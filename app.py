"""Chatterbox Studio: one UI for all models with a persistent voice library and history.

    .\\venv\\Scripts\\python.exe app.py                 # http://127.0.0.1:7860, opens the browser
    .\\venv\\Scripts\\python.exe app.py --port 7861 --no-open
    .\\venv\\Scripts\\python.exe app.py --share        # public Gradio link (off by default)
    .\\venv\\Scripts\\python.exe app.py --check        # build the UI and exit (smoke test)

Voices you upload are kept in ./voices/<name>/ and listed on the left; click one and generate.
Every generation is saved to ./syn_out/history/ and listed in the History tab.
"""
import argparse
import logging
import os
from pathlib import Path

import gradio as gr

from chatterbox.mtl_tts import SUPPORTED_LANGUAGES
from chatterbox.studio import Engine, History, MODEL_SPECS, VoiceLibrary
from chatterbox.studio.engine import GenParams
from chatterbox.studio.library import DEFAULT_VOICE, REPO_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

LIB = VoiceLibrary()
HIST = History()
ENGINE = Engine(LIB)

MODEL_CHOICES = [(spec.label, key) for key, spec in MODEL_SPECS.items()]
LANG_CHOICES = sorted([(f"{name} ({code})", code) for code, name in SUPPORTED_LANGUAGES.items()], key=lambda x: x[0])
PRIORITY_LANGS = ["hi", "en"]
LANG_CHOICES = [c for c in LANG_CHOICES if c[1] in PRIORITY_LANGS] + [c for c in LANG_CHOICES if c[1] not in PRIORITY_LANGS]

EVENT_TAGS = ["[laugh]", "[chuckle]", "[sigh]", "[cough]", "[gasp]", "[groan]", "[sniff]", "[shush]", "[clear throat]"]

SAMPLE_TEXT = {
    "en": "Hi there! Thanks for calling back. [chuckle] Do you have a minute to go over the details?",
    "hi": "नमस्ते! वापस कॉल करने के लिए धन्यवाद। क्या आपके पास विवरण देखने के लिए एक मिनट है?",
}

CSS = """
.voice-list .wrap { max-height: 420px; overflow-y: auto; }
.status-box { font-size: 0.9em; }
footer { display: none !important; }
"""


# ------------------------------------------------------------------ helpers
def voice_choices():
    return [("Default voice (built-in)", DEFAULT_VOICE)] + [(v.label(), v.slug) for v in LIB.list()]


def voice_updates(selected=None):
    """Same choices for every voice picker in the UI."""
    ch = voice_choices()
    values = {c[1] for c in ch}
    sel = selected if selected in values else DEFAULT_VOICE
    return gr.update(choices=ch, value=sel), gr.update(choices=ch, value=sel)


def voice_preview(slug):
    if not slug or slug == DEFAULT_VOICE:
        return None, "**Default voice**: the built-in speaker shipped with each model."
    v = LIB.get(slug)
    if v is None:
        return None, "Voice not found (deleted?). Refresh the list."
    warn = "" if v.turbo_ok else "\n\n⚠️ Shorter than 5 s: usable with Chatterbox / Multilingual only, not Turbo / Nano."
    cached = [p.stem.replace("conds_", "") for p in v.dir.glob("conds_*.pt")] + [p.stem.replace("refdict_", "VC:") for p in v.dir.glob("refdict_*.pt")]
    cached_s = f"\n\nPrepared for: {', '.join(cached)}" if cached else "\n\nPrepared for: (none yet; first use computes and caches)"
    return str(v.ref_path), f"**{v.name}**  ·  {v.duration_s:.1f}s @ {v.sr} Hz  ·  added {v.added}  ·  source `{v.source}`{warn}{cached_s}"


def save_voice(name, audio_path, current):
    if not audio_path:
        raise gr.Error("Upload or record a clip first.")
    try:
        v = LIB.add(audio_path, name)
    except FileExistsError as e:
        raise gr.Error(str(e) + " Delete it first or choose another name.")
    except ValueError as e:
        raise gr.Error(str(e))
    msg = f"Saved **{v.name}** ({v.duration_s:.1f}s)."
    if not v.turbo_ok:
        msg += " Note: under 5 s, so Turbo/Nano will refuse it; Chatterbox and Multilingual work."
    u1, u2 = voice_updates(v.slug)
    return u1, u2, msg, "", None


def delete_voice(slug):
    if not slug or slug == DEFAULT_VOICE:
        raise gr.Error("Select a saved voice to delete (the default voice cannot be deleted).")
    v = LIB.get(slug)
    LIB.delete(slug)
    for key in list(ENGINE._active_voice):
        if ENGINE._active_voice.get(key) == slug:
            ENGINE.apply_voice(key, DEFAULT_VOICE)
    u1, u2 = voice_updates(DEFAULT_VOICE)
    return u1, u2, f"Deleted **{v.name if v else slug}**."


def rename_voice(slug, new_name):
    if not slug or slug == DEFAULT_VOICE:
        raise gr.Error("Select a saved voice to rename.")
    if not new_name.strip():
        raise gr.Error("Enter a new name.")
    v = LIB.rename(slug, new_name)
    u1, u2 = voice_updates(v.slug)
    return u1, u2, f"Renamed to **{v.name}**.", ""


def on_model_change(key, lang, text):
    spec = MODEL_SPECS[key]
    is_sample = (not text.strip()) or text.strip() in SAMPLE_TEXT.values()
    new_text = (SAMPLE_TEXT.get(lang, SAMPLE_TEXT["en"]) if spec.languages else SAMPLE_TEXT["en"]) if is_sample else text
    return (
        gr.update(visible=spec.languages),                       # language dropdown
        gr.update(visible=spec.supports_exaggeration),           # exaggeration
        gr.update(visible=spec.supports_cfg),                    # cfg
        gr.update(visible=spec.supports_min_p),                  # min_p
        gr.update(visible=spec.supports_top_k),                  # top_k
        gr.update(visible=spec.supports_loudness),               # loudness
        gr.update(visible=spec.family == "turbo"),               # tag buttons row
        f"ℹ️ {spec.note}",
        new_text,
    )


def on_lang_change(lang, text):
    if text.strip() in SAMPLE_TEXT.values() or not text.strip():
        return SAMPLE_TEXT.get(lang, text)
    return text


def history_table(limit=100):
    rows = HIST.list(limit)
    table = [[e["ts"], e["model"], e["voice"], e["lang"] or "-", f"{e['duration_s']:.1f}s / {e['gen_s']:.1f}s",
              (e["text"][:90] + "…") if len(e["text"]) > 90 else e["text"]] for e in rows]
    return table, [e["id"] for e in rows]


def generate(model_key, text, voice_slug, lang, exaggeration, cfg, temperature, top_p, min_p, top_k, rep_pen,
             loudness, seed, watermark, split_long, max_chars, gap_ms, progress=gr.Progress()):
    if not text or not text.strip():
        raise gr.Error("Enter some text to synthesize.")
    params = GenParams(exaggeration=exaggeration, cfg_weight=cfg, temperature=temperature, top_p=top_p, min_p=min_p,
                       top_k=int(top_k), repetition_penalty=rep_pen, norm_loudness=loudness, seed=int(seed),
                       watermark=watermark, split_long_text=split_long, max_chars=int(max_chars), gap_ms=int(gap_ms))
    progress(0.0, desc=f"Loading {MODEL_SPECS[model_key].label.split(' (')[0]}…" if model_key not in ENGINE.loaded() else "Preparing voice…")
    try:
        res = ENGINE.generate(model_key, text, voice_slug, lang, params, progress=lambda f, m: progress(f, desc=m))
    except (ValueError, KeyError) as e:
        raise gr.Error(str(e))
    voice_name = "default" if (not voice_slug or voice_slug == DEFAULT_VOICE) else (LIB.get(voice_slug).name if LIB.get(voice_slug) else voice_slug)
    entry = HIST.add(res.wav, res.sr, model=model_key, voice=voice_name, lang=(lang if MODEL_SPECS[model_key].languages else ""),
                     text=text, gen_s=res.gen_s, params=params.as_dict())
    rtf = res.gen_s / max(res.duration_s, 1e-6)
    chunk_info = f"{len(res.chunks)} chunk(s)" if len(res.chunks) > 1 else "1 chunk"
    status = (f"✅ **{res.duration_s:.1f}s** of audio in **{res.gen_s:.1f}s** (RTF {rtf:.2f})  ·  "
              f"{MODEL_SPECS[model_key].label.split(' (')[0]}  ·  voice: {voice_name}  ·  {chunk_info}  ·  "
              f"{'watermarked' if watermark else 'no watermark'}  ·  saved as `{Path(entry.file).name}`")
    table, ids = history_table()
    return entry.file, status, table, ids


def convert_voice(source_path, voice_slug, watermark):
    if not source_path:
        raise gr.Error("Upload the speech you want to convert.")
    try:
        res = ENGINE.convert(source_path, voice_slug, watermark=watermark)
    except (ValueError, KeyError) as e:
        raise gr.Error(str(e))
    voice_name = "default" if (not voice_slug or voice_slug == DEFAULT_VOICE) else LIB.get(voice_slug).name
    entry = HIST.add(res.wav, res.sr, model="vc", voice=voice_name, lang="", text=f"[voice conversion of {Path(source_path).name}]",
                     gen_s=res.gen_s, params={"watermark": watermark})
    table, ids = history_table()
    return entry.file, f"✅ Converted {res.duration_s:.1f}s in {res.gen_s:.1f}s → voice **{voice_name}**", table, ids


def history_select(ids, evt: gr.SelectData):
    if not ids:
        return None, "History is empty."
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row is None or row >= len(ids):
        return None, ""
    e = next((x for x in HIST.list() if x["id"] == ids[row]), None)
    if e is None:
        return None, "Entry no longer exists."
    p = e["params"]
    return e["file"], (f"**{e['ts']}** · {e['model']} · voice {e['voice']} · {e['lang'] or '-'} · {e['duration_s']}s in {e['gen_s']}s\n\n"
                       f"> {e['text']}\n\n`{ {k: p[k] for k in p if k in ('exaggeration','cfg_weight','temperature','top_p','min_p','top_k','repetition_penalty','seed','watermark')} }`")


def models_status():
    lines = []
    for key, spec in MODEL_SPECS.items():
        state = f"loaded ({ENGINE.load_times.get(key, 0):.0f}s)" if key in ENGINE.loaded() else "not loaded"
        lines.append(f"- **{spec.label}** — {state}")
    lines.append(f"- **Voice conversion** — {'loaded' if 'vc' in ENGINE.loaded() else 'not loaded'}")
    import torch
    if torch.cuda.is_available():
        lines.append(f"\nGPU: {torch.cuda.get_device_name(0)} · allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    lines.append(f"\nVoices dir: `{LIB.root}`  ·  History dir: `{HIST.root}`")
    return "\n".join(lines)


def load_all_models(progress=gr.Progress()):
    keys = list(MODEL_SPECS) + ["vc"]
    for i, k in enumerate(keys):
        progress(i / len(keys), desc=f"Loading {k}…")
        ENGINE.get(k)
    return models_status()


def unload_all_models():
    for k in list(ENGINE.loaded()):
        ENGINE.unload(k)
    return models_status()


# ------------------------------------------------------------------ UI
def build():
    with gr.Blocks(title="Chatterbox Studio") as demo:
        gr.Markdown("# 🎙️ Chatterbox Studio\nSaved voices on the left, models and text on the right. Upload a voice once and reuse it with one click.")
        voice_ids_state = gr.State([])

        with gr.Tabs():
            # ------------------------------------------------ Generate tab
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=4, min_width=320):
                        gr.Markdown("### Voices")
                        voice_radio = gr.Radio(choices=voice_choices(), value=DEFAULT_VOICE, label="Click a voice to use it",
                                               elem_classes=["voice-list"])
                        voice_audio = gr.Audio(label="Preview", type="filepath", interactive=False)
                        voice_info = gr.Markdown("**Default voice**: the built-in speaker shipped with each model.")
                        with gr.Row():
                            refresh_voices_btn = gr.Button("🔄 Refresh", size="sm")
                            delete_btn = gr.Button("🗑 Delete voice", size="sm", variant="stop")
                        with gr.Row():
                            rename_box = gr.Textbox(label="Rename to", scale=3)
                            rename_btn = gr.Button("Rename", size="sm", scale=1)
                        voice_msg = gr.Markdown("")
                        with gr.Accordion("➕ Add a voice", open=True):
                            new_name = gr.Textbox(label="Name", placeholder="e.g. Priya, Narrator, Me")
                            new_audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                                 label="Reference clip (6–15 s of clean speech; > 5 s required for Turbo/Nano)")
                            save_btn = gr.Button("💾 Save voice", variant="primary")

                    with gr.Column(scale=8):
                        model_radio = gr.Radio(choices=MODEL_CHOICES, value="turbo", label="Model")
                        model_note = gr.Markdown(f"ℹ️ {MODEL_SPECS['turbo'].note}")
                        lang_dd = gr.Dropdown(choices=LANG_CHOICES, value="hi", label="Language", visible=False)
                        text_box = gr.Textbox(lines=6, label="Text", value=SAMPLE_TEXT["en"],
                                              placeholder="Type or paste text. Long text is split into sentences automatically.")
                        with gr.Row(visible=True) as tag_row:
                            tag_btns = [gr.Button(t, size="sm") for t in EVENT_TAGS]
                        gen_btn = gr.Button("▶ Generate", variant="primary", size="lg")
                        out_audio = gr.Audio(label="Output (24 kHz)", type="filepath", interactive=False)
                        gen_status = gr.Markdown("", elem_classes=["status-box"])

                        with gr.Accordion("⚙️ Settings", open=False):
                            with gr.Row():
                                exaggeration = gr.Slider(0.25, 2.0, value=0.5, step=0.05, label="Exaggeration (0.5 neutral)", visible=False)
                                cfg = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="CFG / pace (0 = off)", visible=False)
                            with gr.Row():
                                temperature = gr.Slider(0.05, 2.0, value=0.8, step=0.05, label="Temperature")
                                top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p (1 = off)")
                                min_p = gr.Slider(0.0, 0.5, value=0.05, step=0.01, label="Min-p (0 = off)", visible=False)
                            with gr.Row():
                                top_k = gr.Slider(0, 1000, value=1000, step=10, label="Top-k (0 = off)", visible=True)
                                rep_pen = gr.Slider(1.0, 2.0, value=1.2, step=0.05, label="Repetition penalty")
                                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                            with gr.Row():
                                loudness = gr.Checkbox(value=True, label="Normalise reference loudness (−27 LUFS)", visible=True)
                                watermark = gr.Checkbox(value=False, label="Apply Perth watermark")
                            with gr.Row():
                                split_long = gr.Checkbox(value=True, label="Split long text into sentences")
                                max_chars = gr.Slider(100, 600, value=300, step=10, label="Max characters per chunk")
                                gap_ms = gr.Slider(0, 1000, value=250, step=10, label="Silence between chunks (ms)")

            # ------------------------------------------------ History tab
            with gr.Tab("History"):
                with gr.Row():
                    with gr.Column(scale=7):
                        table0, ids0 = history_table()
                        hist_table = gr.Dataframe(headers=["When", "Model", "Voice", "Lang", "Audio / gen time", "Text"],
                                                  value=table0, interactive=False, wrap=True, label="Click a row to play")
                        hist_ids = gr.State(ids0)
                        with gr.Row():
                            hist_refresh = gr.Button("🔄 Refresh", size="sm")
                            hist_clear = gr.Button("🗑 Clear all history", size="sm", variant="stop")
                    with gr.Column(scale=5):
                        hist_audio = gr.Audio(label="Playback", type="filepath", interactive=False)
                        hist_info = gr.Markdown("")

            # ------------------------------------------------ Voice conversion tab
            with gr.Tab("Voice conversion"):
                gr.Markdown("Convert any recorded speech into one of your saved voices (content stays, timbre changes). Any language.")
                with gr.Row():
                    with gr.Column():
                        vc_src = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Source speech")
                        vc_voice = gr.Radio(choices=voice_choices(), value=DEFAULT_VOICE, label="Target voice")
                        vc_wm = gr.Checkbox(value=False, label="Apply Perth watermark")
                        vc_btn = gr.Button("🔁 Convert", variant="primary")
                    with gr.Column():
                        vc_out = gr.Audio(label="Converted", type="filepath", interactive=False)
                        vc_status = gr.Markdown("")

            # ------------------------------------------------ Models tab
            with gr.Tab("Models"):
                models_md = gr.Markdown(models_status())
                with gr.Row():
                    load_all_btn = gr.Button("Load all models now (≈ 11 GiB VRAM)")
                    unload_btn = gr.Button("Unload all", variant="stop")
                    models_refresh = gr.Button("Refresh")

        # ------------------------------------------------ wiring
        voice_radio.change(voice_preview, inputs=voice_radio, outputs=[voice_audio, voice_info])
        save_btn.click(save_voice, inputs=[new_name, new_audio, voice_radio],
                       outputs=[voice_radio, vc_voice, voice_msg, new_name, new_audio])
        delete_btn.click(delete_voice, inputs=voice_radio, outputs=[voice_radio, vc_voice, voice_msg])
        refresh_voices_btn.click(lambda cur: voice_updates(cur), inputs=voice_radio, outputs=[voice_radio, vc_voice])
        rename_btn.click(rename_voice, inputs=[voice_radio, rename_box], outputs=[voice_radio, vc_voice, voice_msg, rename_box])

        model_radio.change(on_model_change, inputs=[model_radio, lang_dd, text_box],
                           outputs=[lang_dd, exaggeration, cfg, min_p, top_k, loudness, tag_row, model_note, text_box])
        lang_dd.change(on_lang_change, inputs=[lang_dd, text_box], outputs=text_box)
        for b, t in zip(tag_btns, EVENT_TAGS):
            b.click(lambda text, tag=t: (text.rstrip() + " " + tag + " ").lstrip(), inputs=text_box, outputs=text_box)

        gen_btn.click(generate,
                      inputs=[model_radio, text_box, voice_radio, lang_dd, exaggeration, cfg, temperature, top_p, min_p,
                              top_k, rep_pen, loudness, seed, watermark, split_long, max_chars, gap_ms],
                      outputs=[out_audio, gen_status, hist_table, hist_ids])
        text_box.submit(generate,
                        inputs=[model_radio, text_box, voice_radio, lang_dd, exaggeration, cfg, temperature, top_p, min_p,
                                top_k, rep_pen, loudness, seed, watermark, split_long, max_chars, gap_ms],
                        outputs=[out_audio, gen_status, hist_table, hist_ids])

        hist_table.select(history_select, inputs=hist_ids, outputs=[hist_audio, hist_info])
        hist_refresh.click(lambda: history_table(), outputs=[hist_table, hist_ids])
        hist_clear.click(lambda: (HIST.clear(), history_table())[1], outputs=[hist_table, hist_ids])

        vc_btn.click(convert_voice, inputs=[vc_src, vc_voice, vc_wm], outputs=[vc_out, vc_status, hist_table, hist_ids])

        load_all_btn.click(load_all_models, outputs=models_md)
        unload_btn.click(unload_all_models, outputs=models_md)
        models_refresh.click(models_status, outputs=models_md)

        demo.load(lambda: voice_updates(DEFAULT_VOICE), outputs=[voice_radio, vc_voice])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    ap.add_argument("--check", action="store_true", help="build the UI and exit")
    args = ap.parse_args()
    demo = build()
    if args.check:
        print("UI built OK")
        return
    demo.queue(default_concurrency_limit=1, max_size=20).launch(
        server_name=args.host, server_port=args.port, share=args.share, inbrowser=not args.no_open,
        theme=gr.themes.Soft(), css=CSS, allowed_paths=[str(REPO_ROOT)],
    )


if __name__ == "__main__":
    main()
