import os
import html
import json
import urllib.parse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# APIキーを読み込む（Streamlit Cloud → .env → 環境変数 の優先順）
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
try:
    _gemini_key = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    _gemini_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=_gemini_key)

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


def embed_query(text):
    """ユーザー入力をGemini Embedding APIでベクトル化する"""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=768,
    )
    return np.array(result["embedding"], dtype=np.float32)


def _cosine_similarity(query_vec, doc_matrix):
    """コサイン類似度を計算する（numpy実装）

    Args:
        query_vec: (dim,) のクエリベクトル
        doc_matrix: (n, dim) のドキュメント行列
    Returns:
        (n,) の類似度配列
    """
    query_vec = query_vec.flatten().astype(np.float32)
    doc_matrix = doc_matrix.astype(np.float32)
    dot = doc_matrix @ query_vec
    norm_q = np.linalg.norm(query_vec)
    norm_d = np.linalg.norm(doc_matrix, axis=1)
    denom = norm_q * norm_d
    denom = np.where(denom == 0, 1e-10, denom)
    return dot / denom


@st.cache_data
def load_embeddings():
    """事前計算済みの埋め込みベクトルを読み込む"""
    base_dir = os.path.dirname(__file__)
    emb_dir = os.path.join(base_dir, "data", "embeddings")
    embeddings = {}
    for field in ["ideal_society", "mission", "vision", "values", "description"]:
        path = os.path.join(emb_dir, f"{field}.npy")
        if os.path.exists(path):
            embeddings[field] = np.load(path)
    return embeddings if embeddings else None


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


def compute_match(user_text, df, user_priorities=None, precomputed_embeddings=None, user_skills=""):
    """2段階マッチング: 重み付きベクトル検索で候補絞り込み → Geminiで厳密評価

    Args:
        user_text: ユーザーが入力した理想の社会像テキスト
        df: 企業データフレーム
        user_priorities: ユーザーが選択した重視する観点リスト
        precomputed_embeddings: 事前計算済みの埋め込みベクトル辞書
        user_skills: ユーザーのスキル・経験テキスト
    """
    import re

    user_text_clean = str(user_text).strip()
    user_embedding = embed_query(user_text_clean)

    # Stage 1: フィールド別重み付きベクトル類似度で上位100社を抽出
    field_weights = {
        "ideal_society": 0.35,
        "mission": 0.20,
        "vision": 0.25,
        "values": 0.10,
        "description": 0.10,
    }

    # ユーザーの重視観点に応じて重みを動的調整
    if user_priorities:
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
        total = sum(field_weights.values())
        field_weights = {k: v / total for k, v in field_weights.items()}

    # フィールド別のコサイン類似度を計算して重み付け合成
    weighted_scores = np.zeros(len(df))
    # フィルタ適用時のインデックスマッピング
    df_indices = df.index.tolist()

    for field, weight in field_weights.items():
        if precomputed_embeddings and field in precomputed_embeddings:
            all_embeddings = precomputed_embeddings[field]
            # インデックスが埋め込み配列の範囲内かチェック
            if max(df_indices) < len(all_embeddings):
                field_embeddings = all_embeddings[df_indices]
                sims = _cosine_similarity(user_embedding, field_embeddings)
                texts = df[field].fillna("").tolist()
                non_empty_mask = np.array([bool(t.strip()) for t in texts])
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

    # スキル情報をプロンプトに反映
    skills_instruction = ""
    if user_skills and user_skills.strip():
        skills_instruction = f"""

【ユーザーのスキル・経験】
{user_skills.strip()}
"""

    prompt = f"""あなたは就職マッチングの厳格な審査員です。
ユーザーが実現したい社会像と、以下の{n_candidates}社を比較し、「この企業に入ったらユーザーの理想を本当に実現できるか？」を総合的に評価してください。

【ユーザーの理想の社会像】
{user_text}
{skills_instruction}
【候補企業と各社の目指す社会像・バリュー】
{companies_text}

■ 総合スコアの評価基準（100点満点）:
以下の3つの観点を組み合わせて総合スコアをつけてください。

【観点1: ビジョンの一致度（35%）】
ユーザーの理想と企業が目指す方向性がどれだけ一致しているか。

【観点2: 実現可能性（35%）】
この企業でユーザーの理想を実際に実現できる可能性がどれだけあるか。
以下の要素で判断すること：
- 成長性・将来性（35%）: 新規事業、研究開発、変革への意欲。今後の伸びしろ
- 方向性の本気度（30%）: ビジョンを口だけでなく事業や行動で示しているか
- 挑戦・変革の姿勢（20%）: 既存の枠に囚われず新しいことに挑んでいるか
- 実績・成果（15%）: 過去の取り組みの結果（参考程度）

【観点3: スキル活用度（30%）】
ユーザーが持つスキル・経験をこの企業で活かせるか。
- ユーザーのスキルが企業の事業・ビジョンの実現に直接貢献できるか
- スキルを活かしてユーザーの理想に近づける環境があるか
- スキルが未記入の場合、この観点は無視して観点1と2のみで評価すること

■ 重要な注意点:
- 企業の規模や知名度は評価に含めない。小さくても志が高く成長中の企業を正当に評価すること
- 「社会貢献している」等の抽象表現だけでは高スコアにしない
- 業種が異なっても、目指す社会像が同じなら高スコアをつけてよい
- 安易に高スコアをつけず、本当にマッチし実現可能性もある企業のみ70点以上にする
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
        # Gemini APIエラー時はベクトルスコアのみで評価（フォールバック）
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


def find_relevant_jobs(careers_url, company, skills):
    """採用ページとそのリンク先を読み取り、スキルに関連する求人を探す"""
    import requests as req
    from urllib.parse import urljoin, urlparse

    if not careers_url or not isinstance(careers_url, str) or not careers_url.strip():
        return None, "この企業の採用ページURLが登録されていません。"

    if not skills:
        return None, "スキルが選択されていません。サイドバーからスキルを選択してください。"

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    careers_url = careers_url.strip()

    # ステップ1: 採用トップページからリンクを収集
    try:
        resp = req.get(careers_url, timeout=15, headers=headers, allow_redirects=True)
        resp.encoding = resp.apparent_encoding or "utf-8"
    except Exception:
        return None, f"採用ページにアクセスできませんでした。直接確認してください: [採用ページ]({careers_url})"

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
    except ImportError:
        soup = None

    # リンク抽出
    job_keywords = ["募集", "求人", "職種", "job", "career", "recruit", "entry",
                    "エントリー", "応募", "採用", "新卒", "中途", "position", "openings"]
    found_links = []
    if soup:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            combined = (text + " " + href).lower()
            if any(kw in combined for kw in job_keywords):
                full_url = urljoin(careers_url, href)
                if full_url.startswith("http") and "mailto:" not in href and "#" != href:
                    found_links.append({"text": text[:60], "url": full_url})

    # 重複除去
    seen = set()
    unique_links = []
    for link in found_links:
        if link["url"] not in seen:
            seen.add(link["url"])
            unique_links.append(link)

    # ステップ2: 求人関連リンク先も1階層たどってテキスト収集（最大3ページ）
    sub_pages_text = ""
    for link in unique_links[:3]:
        try:
            sub_resp = req.get(link["url"], timeout=10, headers=headers, allow_redirects=True)
            sub_resp.encoding = sub_resp.apparent_encoding or "utf-8"
            sub_soup = BeautifulSoup(sub_resp.text, "html.parser")
            # テキストとリンクを抽出
            page_text = sub_soup.get_text(separator=" ", strip=True)[:3000]
            sub_links = []
            for a in sub_soup.find_all("a", href=True):
                t = a.get_text(strip=True)
                if t and len(t) > 2:
                    sub_links.append(f"- {t[:50]}: {urljoin(link['url'], a['href'])}")
            sub_pages_text += f"\n\n--- {link['text']} ({link['url']}) ---\n{page_text}\n主なリンク:\n" + "\n".join(sub_links[:20])
        except Exception:
            continue

    # ステップ3: 収集した情報をGeminiに分析させる
    skills_str = ", ".join(skills)
    top_page_text = soup.get_text(separator=" ", strip=True)[:3000] if soup else resp.text[:3000]
    links_summary = "\n".join([f"- {l['text']}: {l['url']}" for l in unique_links[:15]])

    prompt = f"""「{company}」の採用情報から、以下のスキルを持つ人に関連しそうな求人ページを探してください。

【ユーザーのスキル】{skills_str}

【採用トップページ（{careers_url}）の内容】
{top_page_text}

【採用トップページ内のリンク一覧】
{links_summary}

【リンク先ページの内容】
{sub_pages_text}

■ ルール:
- ユーザーのスキルが活かせそうな求人・採用ページのリンクを提示してください
- 完全一致でなくても、関連しそうなものは積極的に含めてください
- 各リンクに「なぜこのスキルと関連しそうか」を一言添えてください
- 最大5件まで
- 見つからない場合でも、一番近そうなページへのリンクを1つは提示してください

■ 出力形式（Markdown）:
- [求人名やページ名](URL) - スキルとの関連理由
"""

    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        response = gemini_model.generate_content(prompt)
        return response.text.strip(), None
    except Exception:
        return None, f"求人情報の分析中にエラーが発生しました。直接確認してください: [採用ページ]({careers_url})"


def generate_explanation(user_text, company, mission, vision, score,
                         values="", ideal_society="", user_priorities=None, user_skills=""):
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
以下のユーザーの理想と企業情報を比較し、以下の2点を具体的に日本語で4〜5文で解説してください。
1. ビジョンの一致度: ユーザーの理想と企業の方向性がどう一致しているか
2. 実現可能性: この企業でユーザーの理想を実現できる可能性はどれくらいか（成長性・挑戦姿勢・本気度を根拠に）
抽象的な表現は避け、企業の事業内容やビジョンの具体的な部分を引用して説明してください。
企業の規模の大小は評価に含めないでください。

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
if "job_results" not in st.session_state:
    st.session_state.job_results = {}
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


def _get_password():
    """合言葉をst.secrets → 環境変数 → デフォルト値の順で取得"""
    try:
        return st.secrets["APP_PASSWORD"]
    except (KeyError, FileNotFoundError):
        return os.getenv("APP_PASSWORD", "Vm#2026$Kibo")


# --- 利用回数制限 ---
_DAILY_LIMIT = 50  # 全ユーザー合計の1日の上限回数
_USAGE_FILE = Path(os.path.dirname(__file__)) / "data" / "usage.json"


def _load_usage():
    """本日の利用回数を読み込む"""
    today = date.today().isoformat()
    if _USAGE_FILE.exists():
        try:
            data = json.loads(_USAGE_FILE.read_text(encoding="utf-8"))
            if data.get("date") == today:
                return data
        except (json.JSONDecodeError, KeyError):
            pass
    return {"date": today, "count": 0, "log": []}


def _save_usage(data):
    """利用回数を保存する"""
    _USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _USAGE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _check_and_increment_usage():
    """利用可能かチェックし、可能なら回数を1増やす。上限超過ならFalseを返す。
    マッチング・AI解説・求人検索すべてを合算してカウントする。"""
    data = _load_usage()
    if data["count"] >= _DAILY_LIMIT:
        return False, data["count"]
    data["count"] += 1
    data["log"].append({
        "time": pd.Timestamp.now().isoformat(),
        "count": data["count"],
    })
    _save_usage(data)
    return True, data["count"]


# --- 合言葉ロック機能 ---
_MAX_LOGIN_ATTEMPTS = 5  # 最大試行回数
_LOCKOUT_MINUTES = 30    # ロック時間（分）
_LOGIN_FILE = Path(os.path.dirname(__file__)) / "data" / "login_attempts.json"


def _load_login_attempts():
    """ログイン試行情報を読み込む"""
    if _LOGIN_FILE.exists():
        try:
            return json.loads(_LOGIN_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            pass
    return {"attempts": 0, "locked_until": None}


def _save_login_attempts(data):
    """ログイン試行情報を保存する"""
    _LOGIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LOGIN_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_locked():
    """ロック中かどうかを確認する"""
    data = _load_login_attempts()
    if data.get("locked_until"):
        locked_until = pd.Timestamp(data["locked_until"])
        if pd.Timestamp.now() < locked_until:
            remaining = (locked_until - pd.Timestamp.now()).seconds // 60
            return True, remaining + 1
        # ロック解除
        _save_login_attempts({"attempts": 0, "locked_until": None})
    return False, 0


def _record_failed_attempt():
    """失敗を記録し、上限に達したらロックする"""
    data = _load_login_attempts()
    data["attempts"] = data.get("attempts", 0) + 1
    if data["attempts"] >= _MAX_LOGIN_ATTEMPTS:
        data["locked_until"] = (pd.Timestamp.now() + pd.Timedelta(minutes=_LOCKOUT_MINUTES)).isoformat()
    _save_login_attempts(data)
    return data["attempts"]


def _reset_login_attempts():
    """ログイン成功時にリセット"""
    _save_login_attempts({"attempts": 0, "locked_until": None})


def _show_login():
    """ログイン画面を表示。認証成功ならTrueを返す"""
    # ログイン画面用CSS
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}

        .login-container {
            max-width: 420px;
            margin: 60px auto 0;
            padding: 40px 36px 32px;
            background: linear-gradient(135deg, #f8faff 0%, #eef2ff 100%);
            border-radius: 16px;
            border: 1px solid #dde4f0;
            box-shadow: 0 4px 24px rgba(30, 58, 95, 0.08);
        }
        .login-logo {
            text-align: center;
            font-size: 2.2em;
            font-weight: 800;
            color: #1e3a5f;
            margin-bottom: 4px;
            letter-spacing: 0.02em;
        }
        .login-sub {
            text-align: center;
            color: #6b7b8f;
            font-size: 0.95em;
            margin-bottom: 28px;
            line-height: 1.6;
        }
        </style>
        <div class="login-container">
            <div class="login-logo">Vision Match</div>
            <div class="login-sub">
                あなたの理想の社会像と<br>企業のビジョンをマッチングするシステムです。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ロック中かチェック
    locked, remaining_min = _is_locked()
    if locked:
        st.error(f"ログイン試行回数の上限に達しました。約{remaining_min}分後に再度お試しください。")
        return False

    # フォームを中央寄せで表示
    _left, center, _right = st.columns([1, 2, 1])
    with center:
        with st.form("login_form"):
            password = st.text_input("パスワード", type="password", placeholder="パスワードを入力してください")
            submitted = st.form_submit_button("ログイン", type="primary", use_container_width=True)
    if submitted:
        if password == _get_password():
            _reset_login_attempts()
            st.session_state.authenticated = True
            st.rerun()
        else:
            attempts = _record_failed_attempt()
            remaining = _MAX_LOGIN_ATTEMPTS - attempts
            if remaining > 0:
                st.error(f"パスワードが正しくありません。（残り{remaining}回）")
            else:
                st.error(f"ログイン試行回数の上限に達しました。{_LOCKOUT_MINUTES}分間ロックされます。")
                st.rerun()
    return False


# --- ログイン認証 ---
if not st.session_state.authenticated:
    _show_login()
    st.stop()

# カスタムCSS
st.markdown(
    """
    <style>
    /* --- 基本レイアウト --- */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* --- マッチ度バッジ --- */
    .match-badge {
        color: white; padding: 6px 16px; border-radius: 24px;
        font-weight: 700; font-size: 1.15em; display: inline-block;
        text-align: center; min-width: 80px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }
    .match-high {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    }
    .match-mid {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }
    .match-low {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }

    /* --- 企業カード --- */
    .company-rank {
        font-size: 1.4em; font-weight: 700; color: #1e3a5f;
        line-height: 1.3;
    }
    .industry-tag {
        background: #e0e7ff; color: #3730a3; padding: 3px 12px;
        border-radius: 12px; font-size: 0.82em; display: inline-block;
        margin-top: 6px; font-weight: 500;
    }
    .card-section-label {
        font-size: 0.85em; font-weight: 600; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.05em;
        margin-bottom: 2px;
    }
    .card-text {
        font-size: 1em; color: #334155; line-height: 1.6;
    }

    /* --- 結果カード内の枠線付きコンテナ --- */
    [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"]
    [data-testid="stHorizontalBlock"] {
        align-items: center;
    }

    /* --- サイドバー --- */
    [data-testid="stSidebar"] h1 {
        font-size: 1.2em;
    }
    [data-testid="stSidebar"] .stDivider {
        margin: 12px 0;
    }

    /* --- ヘッダー部分 --- */
    .app-header {
        text-align: center; padding: 8px 0 4px;
    }
    .app-header h1 {
        font-size: 2em; font-weight: 800; color: #1e3a5f;
        margin-bottom: 0;
    }
    .app-header p {
        color: #64748b; font-size: 1em; margin-top: 4px;
    }

    /* --- 入力エリア --- */
    .input-hint {
        background: #f0f4ff; border-left: 3px solid #3b82f6;
        padding: 10px 14px; border-radius: 0 8px 8px 0;
        font-size: 0.88em; color: #475569; margin-bottom: 12px;
        line-height: 1.6;
    }

    /* --- リンクボタン風 --- */
    .link-row a {
        color: #3b82f6; text-decoration: none; font-weight: 500;
        font-size: 0.9em;
    }
    .link-row a:hover { text-decoration: underline; }

    /* --- フッター --- */
    .app-footer {
        text-align: center; padding: 16px 0 8px;
        color: #94a3b8; font-size: 0.82em;
    }

    /* --- モバイル対応 --- */
    @media (max-width: 768px) {
        .company-rank { font-size: 1.15em; }
        .match-badge { font-size: 1em; padding: 5px 12px; min-width: 64px; }
        .app-header h1 { font-size: 1.5em; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- データ読み込み ---
try:
    df = load_companies()
    precomputed_emb = load_embeddings()
except Exception as e:
    st.error("システムの初期化に失敗しました。ページを再読み込みしてください。")
    st.stop()

# --- サイドバー: スキルタグ（チェックボックス形式） ---
_SKILL_TAGS = {
    "技術系": ["Python", "Java", "Web開発", "AI・機械学習", "データ分析",
               "インフラ・クラウド", "セキュリティ", "モバイルアプリ開発"],
    "ビジネス系": ["営業", "マーケティング", "企画・戦略", "経理・財務",
                  "人事・採用", "法務", "広報・PR", "カスタマーサポート"],
    "クリエイティブ系": ["デザイン・UI/UX", "ライティング・編集", "映像・動画制作"],
    "マネジメント系": ["プロジェクト管理", "チームリーダー", "経営企画", "新規事業開発"],
    "その他": ["研究開発", "コンサルティング", "英語", "中国語"],
}

with st.sidebar:
    st.header("あなたのスキル")
    user_skills_selected = []
    for category, skills in _SKILL_TAGS.items():
        with st.expander(category, expanded=False):
            for skill in skills:
                if st.checkbox(skill, key=f"skill_{skill}"):
                    user_skills_selected.append(skill)

    if user_skills_selected:
        st.caption(f"選択中: {', '.join(user_skills_selected)}")

    st.divider()

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

filtered_df = df.copy()
user_priorities = None
user_skills = ", ".join(user_skills_selected) if user_skills_selected else ""

# --- メインUI ---
st.markdown(
    '<div class="app-header">'
    "<h1>Vision Match</h1>"
    "<p>あなたの理想の社会像と、企業のビジョン・ミッションをAIでマッチングします。</p>"
    "</div>",
    unsafe_allow_html=True,
)
st.divider()

# 入力ガイド
st.markdown(
    '<div class="input-hint">'
    "あなたが将来実現したい社会の姿を、できるだけ具体的に書いてみてください。<br>"
    "「どんな課題を解決したいか」「どんな世界になってほしいか」を意識すると、"
    "より精度の高いマッチングができます。"
    "</div>",
    unsafe_allow_html=True,
)

# 入力フォーム
user_text = st.text_area(
    "あなたが実現したい社会像・ビジョンを入力してください",
    placeholder="例: テクノロジーで社会課題を解決し、誰もが平等にチャンスを得られる世界を作りたい",
    height=130,
)

# 本日の残り回数を表示
_usage = _load_usage()
_remaining = max(0, _DAILY_LIMIT - _usage["count"])
st.caption(f"本日の残り回数: {_remaining} / {_DAILY_LIMIT}")

if st.button("マッチング開始", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("テキストを入力してください。")
    elif len(filtered_df) == 0:
        st.warning("フィルタ条件に該当する企業がありません。サイドバーの条件を変更してください。")
    else:
        allowed, count = _check_and_increment_usage()
        if not allowed:
            st.error(f"本日のマッチング回数が上限（{_DAILY_LIMIT}回）に達しました。明日またお試しください。")
        else:
            try:
                with st.spinner("マッチング中...（候補絞り込み → AI厳密評価、10〜15秒程度）"):
                    st.session_state.results = compute_match(user_text, filtered_df, user_priorities, precomputed_emb, user_skills)
                    st.session_state.user_text = user_text
                    st.session_state.user_priorities = user_priorities
                    st.session_state.explanations = {}
            except Exception:
                st.error("マッチング処理中にエラーが発生しました。時間をおいて再度お試しください。")

# マッチング結果の表示
if st.session_state.results is not None:
    st.divider()

    results_df = st.session_state.results.head(display_count)

    st.subheader(f"マッチング結果（{len(results_df)}件表示）")

    for rank, (i, row) in enumerate(results_df.iterrows(), start=1):
        match_score = row["match_score"]
        company = row["company"]

        # マッチ度に応じたCSSクラス
        if match_score >= 60:
            badge_class = "match-badge match-high"
        elif match_score >= 45:
            badge_class = "match-badge match-mid"
        else:
            badge_class = "match-badge match-low"

        with st.container(border=True):
            head_col, badge_col = st.columns([5, 1])
            with head_col:
                st.markdown(
                    f'<span class="company-rank">{rank}位&ensp;{html.escape(str(company))}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<span class="industry-tag">{html.escape(str(row["industry"]))}</span>',
                    unsafe_allow_html=True,
                )
            with badge_col:
                st.markdown(
                    f'<span class="{badge_class}">{match_score:.1f}%</span>',
                    unsafe_allow_html=True,
                )

            # --- ミッション・ビジョン ---
            mv_col1, mv_col2 = st.columns(2)
            with mv_col1:
                st.markdown('<span class="card-section-label">MISSION</span>', unsafe_allow_html=True)
                st.markdown(f'<p class="card-text">{html.escape(row["mission"] if row["mission"] else "---")}</p>', unsafe_allow_html=True)
            with mv_col2:
                st.markdown('<span class="card-section-label">VISION</span>', unsafe_allow_html=True)
                st.markdown(f'<p class="card-text">{html.escape(row["vision"] if row["vision"] else "---")}</p>', unsafe_allow_html=True)

            # --- ビジョン実現度 ---
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                vs = row.get("vision_score")
                if pd.notna(vs):
                    vs_int = int(vs)
                    st.markdown('<span class="card-section-label">VISION SCORE</span>', unsafe_allow_html=True)
                    st.progress(vs_int / 100, text=f"{vs_int} / 100")
                else:
                    st.markdown('<span class="card-section-label">VISION SCORE</span>', unsafe_allow_html=True)
                    st.caption("---")
            with detail_col2:
                vr = row.get("vision_reason")
                if pd.notna(vr) and str(vr).strip():
                    st.markdown('<span class="card-section-label">SCORE REASON</span>', unsafe_allow_html=True)
                    st.markdown(f'<p class="card-text">{html.escape(str(vr))}</p>', unsafe_allow_html=True)

            # AI解説
            if company in st.session_state.explanations:
                st.info(f"**AI解説:** {st.session_state.explanations[company]}")
            else:
                if st.button("AI解説を見る", key=f"explain_{i}"):
                    allowed, count = _check_and_increment_usage()
                    if not allowed:
                        st.error(f"本日のAPI利用回数が上限（{_DAILY_LIMIT}回）に達しました。明日またお試しください。")
                    else:
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

            # スキルに合う求人を探す
            if user_skills_selected:
                if company in st.session_state.job_results:
                    job_result = st.session_state.job_results[company]
                    if job_result.get("error"):
                        st.warning(f"**求人検索:** {job_result['error']}")
                    else:
                        st.success(f"**あなたのスキルに関連しそうな求人:**")
                        st.markdown(job_result["content"])
                else:
                    if st.button("あなたのスキルに合う求人を探す", key=f"jobs_{i}"):
                        allowed, count = _check_and_increment_usage()
                        if not allowed:
                            st.error(f"本日のAPI利用回数が上限（{_DAILY_LIMIT}回）に達しました。明日またお試しください。")
                        else:
                            with st.spinner("採用ページを確認中..."):
                                content, error = find_relevant_jobs(
                                    row.get("careers_url", ""),
                                    company,
                                    user_skills_selected,
                                )
                            if error:
                                st.session_state.job_results[company] = {"error": error}
                            else:
                                st.session_state.job_results[company] = {"content": content}
                            st.rerun()

            # リンク
            link_cols = st.columns(3)
            with link_cols[0]:
                hp_url = row.get("homepage_url", "")
                if pd.notna(hp_url) and str(hp_url).strip() and str(hp_url).strip().startswith("http"):
                    st.markdown(f"[公式サイト]({hp_url})")
                else:
                    st.caption("公式サイト: ---")
            with link_cols[1]:
                cr_url = row.get("careers_url", "")
                if pd.notna(cr_url) and str(cr_url).strip() and str(cr_url).strip().startswith("http"):
                    st.markdown(f"[採用情報]({cr_url})")
                else:
                    st.caption("採用情報: ---")
            with link_cols[2]:
                ow_search = "https://www.openwork.jp/company_list?src_str=" + urllib.parse.quote(str(row["company"]))
                st.markdown(f"[クチコミ（OpenWork）]({ow_search})")

# フッター
st.markdown(
    '<div class="app-footer">'
    f"登録企業数: {len(df)}社 | 2段階AIマッチング（ベクトル検索 + AI評価）<br>"
    "&copy; 2026 Vision Match All rights reserved."
    "</div>",
    unsafe_allow_html=True,
)
