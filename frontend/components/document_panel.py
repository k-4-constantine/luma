# frontend/components/document_panel.py
from pathlib import Path

def render_document_panel(st):
    """Render the document panel showing retrieved documents."""
    if not st.session_state.retrieved_documents:
        st.info("Retrieved documents will appear here.")
        return

    for idx, doc in enumerate(st.session_state.retrieved_documents, 1):
        relevance_pct = int(doc["relevance_score"] * 100)

        st.markdown(f"""
        <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem;
                    margin-bottom: 1rem; border-left: 4px solid #1f77b4;">
            <div style="font-weight: bold; color: #1f77b4;">
                {idx}. {doc['title']}
            </div>
            <div style="color: #666; font-size: 0.9rem;">
                Relevance: {relevance_pct}% | Type: {doc['file_type'].upper()}
            </div>
            {'<div>Author: ' + doc["author"] + '</div>' if doc.get("author") else ''}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("View Summary"):
            st.write(doc["summary"])

        # Keywords
        if doc.get("keywords"):
            for kw in doc["keywords"]:
                st.markdown(f'<span style="background: #e1e8f0; padding: 0.2rem 0.5rem; border-radius: 0.25rem; margin-right: 0.5rem;">{kw}</span>',
                            unsafe_allow_html=True)

        # File link
        st.markdown(f"[Open File]({Path(doc['file_path']).as_uri()})")
        st.markdown("---")