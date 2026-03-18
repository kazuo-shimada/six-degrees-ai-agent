import gradio as gr
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from pyvis.network import Network
import re
import html

# --- SETUP ---
MODEL_PATH = "/Users/x/Documents/siva_env/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

print("Loading Llama 3 Brain... (Preparing the Multi-Agent Pipeline)")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.7, 
    max_tokens=1200, 
    n_ctx=4096,
    verbose=False
)

wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)

# --- AGENT 1: THE STORYTELLER (High Chaos) ---
story_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert researcher playing "Six Degrees of Separation".
Connect the two topics in exactly 5 to 6 distinct, highly factual steps.
Introduce weird, surprising intermediate concepts to act as bridges.
Write a dramatic, educational story detailing this convoluted chain of events.
Do NOT worry about formatting, just write the story.
<|eot_id|><|start_header_id|>user<|end_header_id|>
RESEARCH FOR TOPIC A:
{context_a}

RESEARCH FOR TOPIC B:
{context_b}

Connect Topic A: {topic_a} to Topic B: {topic_b}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
story_chain = PromptTemplate.from_template(story_template) | llm

# --- AGENT 2: THE CARTOGRAPHER (Strict Extraction) ---
extract_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict data extraction bot. Read the provided story and extract the chronological chain of concepts that connect the starting topic to the ending topic.
Return ONLY a single line starting with "PATH:" followed by the concepts separated by " | ".
Example: PATH: Topic A | Concept 1 | Concept 2 | Concept 3 | Topic B
<|eot_id|><|start_header_id|>user<|end_header_id|>
STORY TO ANALYZE:
{generated_story}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
extract_chain = PromptTemplate.from_template(extract_template) | llm

# --- GRAPH GENERATION ---
def generate_graph_html(path_string):
    nodes = [node.strip() for node in path_string.split('|')]
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="#000000", directed=True)
    
    for i, node in enumerate(nodes):
        color = "#ff4b4b" if i == 0 or i == len(nodes)-1 else "#4b4bff"
        net.add_node(i, label=node, shape="box", color=color)
        if i > 0:
            net.add_edge(i-1, i)
            
    net.save_graph("evidence_board.html")
    with open("evidence_board.html", "r", encoding="utf-8") as f:
        raw_html = f.read()
        
    escaped_html = html.escape(raw_html)
    return f'<iframe srcdoc="{escaped_html}" style="width: 100%; height: 520px; border: none;"></iframe>'

# --- APP LOGIC ---
def find_connection(topic_a, topic_b, chaos_level):
    if not topic_a or not topic_b:
        yield "⚠️ Please enter both topics!", ""
        return
    
    # STEP 1: Research
    yield f"🌐 *Searching Wikipedia for '{topic_a}' and '{topic_b}'...*", ""
    try: wiki_data_a = wiki.run(topic_a)
    except: wiki_data_a = "No data."
    try: wiki_data_b = wiki.run(topic_b)
    except: wiki_data_b = "No data."
    
    # STEP 2: The Storyteller (Uses UI Chaos Level)
    yield f"✍️ *Writing the narrative at Chaos Level {chaos_level}...*", ""
    llm.temperature = chaos_level
    story_response = story_chain.invoke({
        "topic_a": topic_a, "topic_b": topic_b,
        "context_a": wiki_data_a, "context_b": wiki_data_b
    }).strip()
    
    yield story_response, "<p><em>Drawing the Evidence Board...</em></p>"
    
    # STEP 3: The Cartographer (Strictly analytical)
    llm.temperature = 0.1
    extraction_response = extract_chain.invoke({
        "generated_story": story_response
    }).strip()
    
    # STEP 4: Mapping
    path_match = re.search(r'\*?\*?PATH:\*?\*?\s*(.*)', extraction_response, re.IGNORECASE)
    
    if path_match:
        path_string = path_match.group(1)
        graph_html = generate_graph_html(path_string)
        yield story_response, graph_html
    else:
        yield story_response, "<p style='color:red;'>⚠️ The Cartographer failed to map the path. Try again!</p>"

# --- MODERN UI ---
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🔗 Six Degrees of AI Separation")
    gr.Markdown("Enter two completely unrelated topics. Watch the Multi-Agent pipeline research, write, and map the connections.")
    
    with gr.Row():
        t1 = gr.Textbox(label="Starting Point (Topic A)", placeholder="e.g., Black Holes")
        t2 = gr.Textbox(label="Destination (Topic B)", placeholder="e.g., The Mona Lisa")
        
    chaos_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Chaos Level")
    btn = gr.Button("Build the Evidence Board", variant="primary")
    
    with gr.Row():
        with gr.Column(scale=1):
            output_story = gr.Markdown(label="The Narrative")
        with gr.Column(scale=1):
            output_graph = gr.HTML(label="The Evidence Board")
            
    btn.click(fn=find_connection, inputs=[t1, t2, chaos_slider], outputs=[output_story, output_graph])

if __name__ == "__main__":
    demo.launch()
