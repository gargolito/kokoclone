import gradio as gr
import os
from core.cloner import KokoClone

# 1. Initialize the cloner globally so models load only once when the server starts
print("Loading KokoClone models for the Web UI...")
cloner = KokoClone()

def clone_voice(text, lang, ref_audio_path):
    """Gradio prediction function."""
    if not text or not text.strip():
        raise gr.Error("Please enter some text.")
    if not ref_audio_path:
        raise gr.Error("Please upload or record a reference audio file.")
    
    output_file = "gradio_output.wav"
    
    try:
        # Call the core engine
        cloner.generate(
            text=text,
            lang=lang,
            reference_audio=ref_audio_path,
            output_path=output_file
        )
        return output_file
    except Exception as e:
        raise gr.Error(f"An error occurred during generation: {str(e)}")

# 2. Build the Gradio UI using Blocks
# Gradio 6.0 fix: Removed theme from here
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>🎧 KokoClone</h1>
            <p>Voice Cloning, Now Inside Kokoro.<br>
            Generate natural multilingual speech and clone any target voice with ease.<br>
            <i>Built on Kokoro TTS.</i></p>
        </div>
        """
    )
    
    with gr.Row():
        # LEFT COLUMN: Inputs
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="1. Text to Synthesize", 
                lines=4, 
                placeholder="Enter the text you want spoken..."
            )
            
            lang_input = gr.Dropdown(
                label="2. Language",
                choices=[
                    ("English", "en"), 
                    ("Hindi", "hi"), 
                    ("French", "fr"), 
                    ("Japanese", "ja"), 
                    ("Chinese", "zh"), 
                    ("Italian", "it"), 
                    ("Spanish", "es"), 
                    ("Portuguese", "pt")
                ],
                value="en"
            )
            
            # Using type="filepath" passes the temp file path directly to our cloner
            ref_audio_input = gr.Audio(
                label="3. Reference Voice (Upload or Record)", 
                type="filepath" 
            )
            
            submit_btn = gr.Button("🚀 Generate Clone", variant="primary")
            
        # RIGHT COLUMN: Outputs and Instructions
        with gr.Column(scale=1):
            output_audio = gr.Audio(
                label="Generated Cloned Audio", 
                interactive=False, 
                autoplay=False
            )
            
            gr.Markdown(
                """
                <br>
                
                ### 💡 Tips for Best Results:
                * **Clean Audio:** Use a reference audio clip without background noise or music.
                * **Length:** A reference clip of 3 to 10 seconds is usually the sweet spot.
                * **Language Match:** Make sure the selected language matches the text you typed!
                * **First Run:** The very first generation might take a few extra seconds while the models allocate memory.
                """
            )

    # 3. Wire the button to the function
    submit_btn.click(
        fn=clone_voice,
        inputs=[text_input, lang_input, ref_audio_input],
        outputs=output_audio
    )

# 4. Launch the app
if __name__ == "__main__":
    # Gradio 6.0 fix: Moved theme here and removed show_api
    demo.launch()