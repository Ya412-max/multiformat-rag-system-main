import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG System",
    page_icon="",
    layout="wide",
)


def init_session_state():
    defaults = {
        "messages": [],
        "uploaded_files": [],
        "top_k": 5,
        "enable_hybrid": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def upload_file(uploaded_file):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        resp = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=120)

        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Uploaded: {data['filename']} → {data['total_chunks']} chunks")
            return True
        else:
            detail = resp.json().get("detail", resp.text)
            st.error(f"Upload failed: {detail}")
            return False
    except requests.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE_URL}")
        return False
    except Exception as e:
        st.error(f"Upload error: {e}")
        return False


def send_question(question: str):
    st.session_state["messages"].append({"role": "user", "content": question})
    try:
        payload = {
            "question": question,
            "top_k": st.session_state["top_k"],
            "stream": False,
            "enable_hybrid": st.session_state["enable_hybrid"],
        }
        resp = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=120)

        if resp.status_code == 200:
            data = resp.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])

            response_text = answer
            if sources:
                response_text += "\n\n---\n**Sources:**\n"
                for s in sources:
                    src = s.get("source", "unknown")
                    page = s.get("page", "")
                    page_info = f" (Page {page})" if page else ""
                    response_text += f"\n- {src}{page_info}"

            st.session_state["messages"].append({"role": "assistant", "content": response_text})
        else:
            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"Error: {resp.status_code} - {resp.text}"
            })
    except requests.ConnectionError:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": f"Error: Cannot connect to API server."
        })
    except Exception as e:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": f"Error: {e}"
        })


def fetch_documents():
    try:
        resp = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def main():
    init_session_state()
    fetch_documents()

    with st.sidebar:
        st.title("RAG System")

        st.subheader("Upload Document")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=["pdf", "docx", "doc", "txt", "md", "html", "htm", "xlsx", "xls", "csv", "png", "jpg", "jpeg"],
        )
        if uploaded_files:
            for uf in uploaded_files:
                fname = uf.name
                if fname not in st.session_state["uploaded_files"]:
                    with st.spinner(f"Processing {fname}..."):
                        success = upload_file(uf)
                    if success:
                        st.session_state["uploaded_files"].append(fname)
                        st.rerun()

        st.subheader("Settings")
        st.session_state["top_k"] = st.slider("Top K", 1, 20, 5)
        st.session_state["enable_hybrid"] = st.checkbox("Hybrid Search", value=True)

        st.subheader("Documents")
        docs = fetch_documents()
        if docs:
            for idx, doc in enumerate(docs):
                col1, col2 = st.columns([4, 1])
                file_path = doc.get("file_path", "")
                with col1:
                    st.text(f"{doc['source']} ({doc['file_type']})")
                with col2:
                    delete_clicked = st.button("Delete", key=f"del_btn_{idx}")
                    if delete_clicked:
                        try:
                            resp = requests.delete(
                                f"{API_BASE_URL}/documents",
                                params={"file_path": file_path},
                                timeout=10,
                            )
                            res_data = resp.json()
                            deleted_num = res_data.get("deleted_chunks", 0)
                            if deleted_num > 0:
                                st.success(f"删除成功，移除{deleted_num}个分片")
                                st.rerun()
                            else:
                                st.warning("未找到对应分片，数据没有被删除")
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
                            st.rerun()
        else:
            st.info("No documents uploaded")

        if st.button("Clear Chat History"):
            st.session_state["messages"] = []
            st.rerun()

    st.title("RAG Chat")
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question...")
    if question:
        send_question(question)
        st.rerun()


if __name__ == "__main__":
    main()




