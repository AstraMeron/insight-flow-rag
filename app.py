import gradio as gr
import rag_logic
import time

def predict(message, history):
    try:
        context, docs, meta = rag_logic.get_relevant_context(message)
        answer = rag_logic.generate_rag_response(message, context)

        # STREAM THE ANSWER FIRST
        buffer = ""
        for char in answer:
            buffer += char
            yield buffer
            time.sleep(0.003)

        # BUILD SOURCES (non-streamed)
        source_parts = []
        if docs:
            for i, d in enumerate(docs[:3]):
                text_content = str(d).strip()
                if text_content:
                    source_parts.append(
                        f"**Source {i+1}:**\n> {text_content[:400]}..."
                    )

        if source_parts:
            sources_text = (
                "\n\n---\n"
                "### 📂 Sources Used for Verification:\n\n"
                + "\n\n".join(source_parts)
            )
        else:
            sources_text = (
                "\n\n---\n"
                "### 📂 Sources Used for Verification:\n\n"
                "_No relevant records found._"
            )

        # SEND SOURCES AS A SECOND MESSAGE
        yield answer + sources_text

    except Exception as e:
        yield f"⚠️ **System Error:** {str(e)}"


# --- UI Setup: Fully Compliant with Task 4 Requirements ---
with gr.Blocks(title="CrediTrust Analyst Portal") as demo:
    gr.Markdown("# 🏦 CrediTrust Intelligence Portal")
    gr.Markdown("Interactive RAG system for analyzing customer complaint data.")
    
    # ChatInterface provides the Text Input, Submit, and Clear buttons
    gr.ChatInterface(
        fn=predict,
        examples=[
            "What are common credit card billing disputes?", 
            "Are there fees for closing accounts?",
            "How does the bank handle unauthorized transactions?"
        ],
        fill_height=True
    )

if __name__ == "__main__":
    # Theme moved to launch as per Gradio 6.0 standards
    demo.launch(theme="soft", share=True)
    