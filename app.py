import os
import urllib.parse
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# .envファイルからAPIキーを読み込む
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ページ設定
st.set_page_config(
    page_title="ビジョンマッチ - 企業理念マッチングシステム",
    page_icon="🏢",
    layout="wide",
    menu_items={
        "Get help": None,
        "Report a Bug": None,
        "About": "ビジョンマッチ - あなたの理想と企業のビジョンをマッチングするシステムです。",
    },
)


@st.cache_resource
def load_model():
    """モデルをキャッシュして読み込む（初回のみダウンロード）"""
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data
def load_companies():
    """企業データを読み込む（理想の社会像データがあればマージ）"""
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "companies.csv")
    df = pd.read_csv(csv_path, encoding="utf-8")
    for col in ["mission", "vision", "values", "description"]:
        df[col] = df[col].fillna("")

    # 理想の社会像データをマージ
    ideal_path = os.path.join(base_dir, "ideal_society.csv")
    if os.path.exists(ideal_path):
        ideal_df = pd.read_csv(ideal_path, encoding="utf-8")
        df = df.merge(ideal_df[["company", "ideal_society"]], on="company", how="left")
        df["ideal_society"] = df["ideal_society"].fillna("")
    else:
        df["ideal_society"] = ""

    # マッチング用テキスト: 理想の社会像があればそれを優先、なければMVV
    df["combined_text"] = df.apply(
        lambda r: r["ideal_society"] if r["ideal_society"] else r["mission"] + " " + r["vision"],
        axis=1,
    )
    return df


def compute_match(user_text, df, model, user_priorities=None):
    """2段階マッチング: 重み付きベクトル検索で候補絞り込み → Geminiで厳密評価

    Args:
        user_text: ユーザーが入力した理想の社会像テキスト
        df: 企業データフレーム
        model: SentenceTransformerモデル
        user_priorities: ユーザーが選択した重視する観点リスト
    """
    import re
    import numpy as np

    user_text_clean = str(user_text).strip()
    user_embedding = model.encode([user_text_clean])

    # Stage 1: フィールド別重み付きベクトル類似度で上位100社を抽出
    # 各フィールドの埋め込みを個別に計算
    field_weights = {
        "ideal_society": 0.35,
        "mission": 0.20,
        "vision": 0.25,
        "values": 0.10,
        "description": 0.10,
    }

    # ユーザーの重視観点に応じて重みを動的調整
    if user_priorities:
        # 重視する観点のキーワードとフィールドの対応
        priority_boost = {
            "社会貢献": {"ideal_society": 0.10, "mission": 0.05},
            "技術革新": {"description": 0.10, "vision": 0.05},
            "働きやすさ": {"values": 0.15},
            "成長性": {"vision": 0.10, "description": 0.05},
            "理念共感": {"mission": 0.10, "ideal_society": 0.05},
        }
        for priority in user_priorities:
            if priority in priority_boost:
                for field, boost in priority_boost[priority].items():
                    field_weights[field] = field_weights.get(field, 0) + boost
        # 正規化
        total = sum(field_weights.values())
        field_weights = {k: v / total for k, v in field_weights.items()}

    # フィールド別のコサイン類似度を計算して重み付け合成
    weighted_scores = np.zeros(len(df))
    for field, weight in field_weights.items():
        texts = df[field].fillna("").tolist()
        # 空テキストはスコア0として扱う
        non_empty_mask = np.array([bool(t.strip()) for t in texts])
        if non_empty_mask.any():
            embeddings = model.encode(texts)
            sims = cosine_similarity(user_embedding, embeddings)[0]
            # 空テキストのスコアは0
            sims[~non_empty_mask] = 0
            weighted_scores += sims * weight

    results = df[["company", "industry", "mission", "vision", "values", "description",
                   "homepage_url", "careers_url",
                   "vision_score", "vision_reason", "ideal_society"]].copy()
    results["vector_score"] = (weighted_scores * 100).round(1).astype(float)

    # vision_score（ビジョン実現度）をベクトルスコアに加味（最大+5点のボーナス）
    vs = pd.to_numeric(results["vision_score"], errors="coerce").fillna(0)
    results["vector_score"] = results["vector_score"] + (vs / 100 * 5).round(1)

    top_candidates = results.sort_values("vector_score", ascending=False).head(100)
    n_candidates = len(top_candidates)

    # Stage 2: Geminiで厳密に再評価
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    companies_text = ""
    for idx, (_, row) in enumerate(top_candidates.iterrows()):
        ideal = row.get("ideal_society", "")
        if not ideal or (isinstance(ideal, float)) or not str(ideal).strip():
            ideal = " / ".join(filter(None, [row["mission"], row["vision"]]))
        values_text = row.get("values", "")
        if values_text and isinstance(values_text, str) and values_text.strip():
            ideal += f"【バリュー】{values_text}"
        companies_text += f"{idx+1}. {row['company']}（{row['industry']}）: {ideal}\n"

    # ユーザーの重視観点をプロンプトに反映
    priority_instruction = ""
    if user_priorities:
        priority_instruction = f"""
■ ユーザーが特に重視する観点: {', '.join(user_priorities)}
→ 上記の観点に合致する企業は加点して評価してください。"""

    prompt = f"""あなたは就職マッチングの厳格な審査員です。
ユーザーが実現したい社会像と、以下の{n_candidates}社が目指す社会像を比較し、本当に一致している企業だけに高スコアをつけてください。

【ユーザーの理想の社会像】
{user_text}
{priority_instruction}

【候補企業と各社の目指す社会像・バリュー】
{companies_text}

■ 評価基準（100点満点）:
- 90-100点: ユーザーの理想とほぼ同じ社会を目指している（方向性・具体的な手段が一致）
- 70-89点: 大きな方向性が一致し、事業を通じて直接的に貢献している
- 50-69点: 部分的に関連するが、ユーザーの理想とは異なる側面もある
- 30-49点: 間接的に関連する程度。ユーザーの理想との直接的なつながりは薄い
- 0-29点: ほぼ無関係

■ 評価の注意点:
- 「社会貢献している」「世界をより良くする」等の抽象的な表現だけでは高スコアにしない
- ユーザーの理想の「具体的な内容」と企業の「具体的な事業・方向性」が一致しているかを厳密に判定
- 業種が異なっても、目指す社会像が同じなら高スコアをつけてよい
- 安易に高スコアをつけず、本当にマッチしている企業のみ70点以上にする
- 企業のバリュー（行動指針）がユーザーの理想と合致する場合は加点要素とする
- 平均スコアが50を超えないよう、厳格に採点すること

以下の形式で{n_candidates}社すべてのスコアを出力してください（1行1社、番号とスコアのみ）:
1: スコア
2: スコア
...
{n_candidates}: スコア"""

    ai_scores = {}
    try:
        response = gemini_model.generate_content(prompt)
        for line in response.text.strip().split("\n"):
            line = line.strip()
            match = re.match(r"(\d+)\s*[:：]\s*(\d+)", line)
            if match:
                idx = int(match.group(1))
                score = max(0, min(100, int(match.group(2))))
                ai_scores[idx] = score
    except Exception:
        pass

    # AIスコアとvision_scoreを組み合わせた最終スコア
    top_list = top_candidates.reset_index(drop=True)
    match_scores = []
    for idx in range(len(top_list)):
        ai_score = ai_scores.get(idx + 1, None)
        if ai_score is not None:
            # vision_score（ビジョン実現度）を最終スコアに反映（最大+3点ボーナス）
            vs_val = pd.to_numeric(top_list.iloc[idx].get("vision_score"), errors="coerce")
            vs_bonus = (vs_val / 100 * 3) if pd.notna(vs_val) else 0
            final_score = min(100, ai_score + vs_bonus)
            match_scores.append(round(final_score, 1))
        else:
            # AIスコアが取れなかった場合はベクトルスコアをフォールバック
            match_scores.append(top_list.iloc[idx]["vector_score"])
    top_list["match_score"] = match_scores
    top_list = top_list.sort_values("match_score", ascending=False).reset_index(drop=True)
    return top_list


def generate_explanation(user_text, company, mission, vision, score,
                         values="", ideal_society="", user_priorities=None):
    """Gemini APIで1社分のマッチング理由を生成"""
    priority_text = ""
    if user_priorities:
        priority_text = f"\n【ユーザーが重視する観点】{', '.join(user_priorities)}"

    ideal_text = ""
    if ideal_society and isinstance(ideal_society, str) and ideal_society.strip():
        ideal_text = f"\n【企業が目指す社会像】{ideal_society}"

    values_text = ""
    if values and isinstance(values, str) and values.strip():
        values_text = f"\n【バリュー】{values}"

    prompt = f"""あなたは就職支援の専門アドバイザーです。
以下のユーザーの理想と企業情報を比較し、マッチしている点・していない点を具体的に日本語で3〜4文で解説してください。
抽象的な表現は避け、企業の事業内容やビジョンの具体的な部分を引用して説明してください。

【ユーザーの理想の社会像】
{user_text}{priority_text}

【企業名】{company}
【ミッション】{mission}
【ビジョン】{vision}{ideal_text}{values_text}
【マッチ度】{score:.1f}%

解説:"""

    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"（解説を生成できませんでした: {e}）"


# セッション状態の初期化
if "results" not in st.session_state:
    st.session_state.results = None
if "user_text" not in st.session_state:
    st.session_state.user_text = ""
if "explanations" not in st.session_state:
    st.session_state.explanations = {}
if "user_priorities" not in st.session_state:
    st.session_state.user_priorities = []

# カスタムCSS
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* マッチ度バッジ */
    .match-high {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white; padding: 4px 12px; border-radius: 20px;
        font-weight: bold; font-size: 1.1em; display: inline-block;
    }
    .match-mid {
        background: linear-gradient(135deg, #eab308, #ca8a04);
        color: white; padding: 4px 12px; border-radius: 20px;
        font-weight: bold; font-size: 1.1em; display: inline-block;
    }
    .match-low {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white; padding: 4px 12px; border-radius: 20px;
        font-weight: bold; font-size: 1.1em; display: inline-block;
    }

    /* 企業カード */
    .company-rank {
        font-size: 1.5em; font-weight: bold; color: #1e3a5f;
    }
    .industry-tag {
        background: #e0e7ff; color: #3730a3; padding: 2px 10px;
        border-radius: 12px; font-size: 0.85em; display: inline-block;
        margin-top: 4px;
    }

    /* サイドバータイトル */
    [data-testid="stSidebar"] h1 {
        font-size: 1.2em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- モデル・データ読み込み ---
with st.spinner("モデルを読み込み中..."):
    model = load_model()
df = load_companies()

# --- サイドバー: フィルタ・検索 ---
with st.sidebar:
    st.header("検索・フィルタ設定")

    st.subheader("企業名で検索")
    company_search = st.text_input(
        "企業名（部分一致）",
        placeholder="例: トヨタ",
        label_visibility="collapsed",
    )

    st.subheader("業種フィルタ")
    all_industries = sorted(df["industry"].dropna().unique().tolist())
    # 「-」など無効な値を除外
    all_industries = [ind for ind in all_industries if ind.strip() and ind.strip() != "-"]
    selected_industries = st.multiselect(
        "業種を選択（複数可）",
        options=all_industries,
        default=[],
        placeholder="すべての業種を表示中...",
        label_visibility="collapsed",
    )

    st.subheader("重視する観点")
    user_priorities = st.multiselect(
        "マッチングで重視する観点を選択（複数可）",
        options=["社会貢献", "技術革新", "働きやすさ", "成長性", "理念共感"],
        default=[],
        placeholder="選択なし（標準の重み付け）",
        label_visibility="collapsed",
    )

    st.subheader("表示件数")
    display_count = st.radio(
        "マッチング結果の表示件数",
        options=[10, 20, 50],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"登録企業数: {len(df)}社")

    # フィルタ適用後の企業数を表示
    filtered_df = df.copy()
    if company_search.strip():
        filtered_df = filtered_df[
            filtered_df["company"].str.contains(company_search.strip(), case=False, na=False)
        ]
    if selected_industries:
        filtered_df = filtered_df[filtered_df["industry"].isin(selected_industries)]
    if len(filtered_df) != len(df):
        st.caption(f"フィルタ適用後: {len(filtered_df)}社")

# --- メインUI ---
st.title("ビジョンマッチ")
st.markdown("あなたの理想の社会像と、企業のビジョン・ミッションをマッチングします。")
st.divider()

# 入力フォーム
user_text = st.text_area(
    "あなたが実現したい社会像・ビジョンを入力してください",
    placeholder="例: テクノロジーで社会課題を解決し、誰もが平等にチャンスを得られる世界を作りたい",
    height=120,
)

if st.button("マッチング開始", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("テキストを入力してください。")
    elif len(filtered_df) == 0:
        st.warning("フィルタ条件に該当する企業がありません。サイドバーの条件を変更してください。")
    else:
        with st.spinner("マッチング中...（候補絞り込み → AI厳密評価、10〜15秒程度）"):
            st.session_state.results = compute_match(user_text, filtered_df, model, user_priorities)
            st.session_state.user_text = user_text
            st.session_state.user_priorities = user_priorities
            st.session_state.explanations = {}

# マッチング結果の表示
if st.session_state.results is not None:
    st.divider()

    results_df = st.session_state.results

    # 結果にもサイドバーのフィルタを適用（マッチング後に条件変更した場合）
    if company_search.strip():
        results_df = results_df[
            results_df["company"].str.contains(company_search.strip(), case=False, na=False)
        ]
    if selected_industries:
        results_df = results_df[results_df["industry"].isin(selected_industries)]

    results_df = results_df.head(display_count)

    st.subheader(f"マッチング結果（{len(results_df)}件表示）")

    for rank, (i, row) in enumerate(results_df.iterrows(), start=1):
        match_score = row["match_score"]
        company = row["company"]

        # マッチ度に応じたCSSクラス
        if match_score >= 60:
            badge_class = "match-high"
        elif match_score >= 45:
            badge_class = "match-mid"
        else:
            badge_class = "match-low"

        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f'<span class="company-rank">{rank}位: {company}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<span class="industry-tag">{row["industry"]}</span>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f'<span class="{badge_class}">{match_score:.1f}%</span>',
                    unsafe_allow_html=True,
                )

            # ミッション・ビジョンを2カラムで表示
            mv_col1, mv_col2 = st.columns(2)
            with mv_col1:
                st.markdown(f"**ミッション**")
                st.caption(row["mission"] if row["mission"] else "---")
            with mv_col2:
                st.markdown(f"**ビジョン**")
                st.caption(row["vision"] if row["vision"] else "---")

            # ビジョン実現度
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                vs = row.get("vision_score")
                if pd.notna(vs):
                    st.metric("ビジョン実現度", f"{int(vs)} / 100")
                else:
                    st.metric("ビジョン実現度", "---")
            with detail_col2:
                vr = row.get("vision_reason")
                if pd.notna(vr) and str(vr).strip():
                    st.markdown("**評価理由**")
                    st.caption(str(vr))

            # AI解説
            if company in st.session_state.explanations:
                st.info(f"**AI解説:** {st.session_state.explanations[company]}")
            else:
                if st.button("AI解説を見る", key=f"explain_{i}"):
                    with st.spinner("AI解説を生成中..."):
                        explanation = generate_explanation(
                            st.session_state.user_text, company,
                            row["mission"], row["vision"], match_score,
                            values=row.get("values", ""),
                            ideal_society=row.get("ideal_society", ""),
                            user_priorities=st.session_state.user_priorities,
                        )
                    st.session_state.explanations[company] = explanation
                    st.rerun()

            # リンク
            link_cols = st.columns(3)
            with link_cols[0]:
                st.markdown(f"[公式サイト]({row['homepage_url']})")
            with link_cols[1]:
                st.markdown(f"[採用情報]({row['careers_url']})")
            with link_cols[2]:
                ow_search = "https://www.openwork.jp/company_list?src_str=" + urllib.parse.quote(row["company"])
                st.markdown(f"[クチコミ（OpenWork）]({ow_search})")

# フッター
st.divider()
st.caption(f"登録企業数: {len(df)}社 | 重み付きベクトル検索（Top100） + AI厳密評価の2段階マッチング")
