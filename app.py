import streamlit as st
import os
import time
import sys
import json
import threading

from pathlib import Path

# === LangChain 相關引用 ===
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings  # 建議使用新版 package

# PDF 讀取與文字切分工具
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Streamlit runtime（本機 exe 用；雲端預設不啟用）
from streamlit.runtime import get_instance


# ==========================================
# 0. 路徑修正：同時支援本機/打包/Streamlit Cloud
# ==========================================
def get_resource_path(relative_path: str) -> str:
    """取得資源檔案的絕對路徑（支援 exe 與 Streamlit Cloud）"""
    if hasattr(sys, "_MEIPASS"):  # PyInstaller 打包情境
        return os.path.join(sys._MEIPASS, relative_path)
    base_dir = Path(__file__).resolve().parent
    return str(base_dir / relative_path)


# ==========================================
# 1. 設定與初始化（Cloud 版：Key 改讀 secrets/env）
# ==========================================
# ✅ Streamlit Community Cloud：請在 App 的 Secrets 填入 GOOGLE_API_KEY
#    或本機用環境變數 GOOGLE_API_KEY
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, "secrets") else None
GOOGLE_API_KEY = GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ 找不到 GOOGLE_API_KEY：請於 Streamlit Cloud 的 Secrets 設定，或在本機設定環境變數。")
    st.stop()

st.set_page_config(page_title="生態檢核小幫手", page_icon="🌿")
st.title("🌿 生態檢核智慧助手 (RAG版)")
st.caption("已掛載：FAQ.json + PDF資料庫 | 模式：專業回答 (Gemini 2.5 Flash)")


# ==========================================
# 2. （可選）本機 exe 專用：無連線自動關機監控
#    Cloud 預設關閉，避免在雲端誤殺/不相容
# ==========================================
ENABLE_AUTO_SHUTDOWN = os.environ.get("ENABLE_AUTO_SHUTDOWN", "0") == "1"

@st.cache_resource
def start_shutdown_monitor():
    """
    啟動背景執行緒：若偵測無連線一段時間則關閉程式（僅建議本機 exe）
    """
    def monitor_loop():
        time.sleep(5)  # 啟動緩衝
        while True:
            try:
                runtime = get_instance()
                # ⚠️ 以下使用私有屬性，可能受版本影響；所以只在 ENABLE_AUTO_SHUTDOWN 時才會執行
                if runtime and hasattr(runtime, "_session_mgr"):
                    session_manager = runtime._session_mgr
                    if hasattr(session_manager, "list_active_sessions"):
                        active_sessions = session_manager.list_active_sessions()
                        count = len(active_sessions)
                    else:
                        count = 1
                else:
                    count = 1  # 取不到就假設有人，避免誤關

                if count == 0:
                    print("👋 偵測到無連線，準備關閉...")
                    time.sleep(3)
                    runtime2 = get_instance()
                    if runtime2 and hasattr(runtime2, "_session_mgr") and hasattr(runtime2._session_mgr, "list_active_sessions"):
                        if len(runtime2._session_mgr.list_active_sessions()) == 0:
                            print("🛑 執行關閉程序。")
                            os._exit(0)

            except Exception as e:
                print(f"Monitor Error: {e}")

            time.sleep(2)

    thread = threading.Thread(target=monitor_loop, daemon=True)
    thread.start()
    return True

if ENABLE_AUTO_SHUTDOWN:
    start_shutdown_monitor()

# ==========================================
# 3. 建立 RAG 大腦（改為：從 GitHub 檔案載入預建索引）
# ==========================================
@st.cache_resource
def init_rag_system(api_key: str):
    # A. 初始化與 build_index.py 相同的 Embedding 模型
    # 注意：此步驟在 Streamlit Cloud 執行不消耗 Google API 額度
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # B. 指定預建索引的資料夾路徑
    index_path = get_resource_path("faiss_index")

    if not os.path.exists(index_path):
        st.error(f"❌ 找不到預建索引資料夾：{index_path}。請確認已將 faiss_index 推送至 GitHub。")
        st.stop()

    # C. 直接載入索引（秒開，不需重新計算）
    try:
        vector_db = FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True  # 載入本地 pkl 檔案需開啟此項
        )
    except Exception as e:
        st.error(f"❌ 載入索引失敗：{e}")
        st.stop()

    # D. 設定檢索器與 LLM
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", google_api_key=api_key)

    return retriever, llm

# 初始化系統
try:
    with st.spinner("正在快速載入生態知識庫索引..."):
        retriever, llm = init_rag_system(GOOGLE_API_KEY)
    st.success("✅ 知識庫已載入！系統已就緒。")
except Exception as e:
    st.error(f"❌ 系統啟動失敗：{e}")
    st.stop()


# ==========================================
# 4. 定義 Prompt 與 RAG Chain
#    ✅ 避免重複檢索：先 retriever.invoke 一次，context 再送給 llm
# ==========================================
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

template = """
你是一位政府水利單位的專業生態檢核顧問。以下是從內部資料彙整的【背景資訊】。
這些資訊可能來自 FAQ 問答集，也可能來自 PDF 規範文件。
以 FAQ 問答集的內容做為優先資料來源。

請依據這些資訊來回答使用者的問題。

### 回答守則：
1. **基於事實**：請優先依據【背景資訊】的內容進行回答。
2. **整合資訊**：如果答案分散在不同段落，請將其整合成通順的說明。
3. **自然專業**：以專業顧問的口吻說明，不需要提到「資料庫」或「文件切片」等字眼。
4. **資訊不足**：如果【背景資訊】與問題無關，請直接回答：「抱歉，目前的資料中暫無此問題的詳細說明，建議查閱原始計畫書件。」

### 背景資訊：
{context}

### 使用者問題：
{question}
"""
prompt = PromptTemplate.from_template(template)
answer_chain = prompt | llm | StrOutputParser()


# ==========================================
# 5. 聊天介面
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史訊息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 輸入區
if user_input := st.chat_input("請輸入生態檢核相關的問題..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 正在檢索 FAQ 與 PDF 資料...")

        try:
            # ✅ 只檢索一次
            source_documents = retriever.invoke(user_input)

            # 顯示檢索結果
            with st.expander("🔍 查看檢索到的原始資料"):
                for idx, doc in enumerate(source_documents):
                    source_name = doc.metadata.get("source", "未知來源")
                    page_num = doc.metadata.get("page", "N/A")
                    st.markdown(f"**文件 {idx+1} (來源: {source_name}, 頁碼: {page_num}):**")
                    st.text(doc.page_content)
                    st.divider()

            context_text = format_docs(source_documents)
            
            # --- 方案A 修改處：改用串流輸出 ---
            message_placeholder.empty() # 清除檢索中的文字
            
            # 使用 st.write_stream 處理 answer_chain.stream
            full_response = st.write_stream(
                answer_chain.stream({"context": context_text, "question": user_input})
            )

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            message_placeholder.error(f"發生錯誤：{e}")
