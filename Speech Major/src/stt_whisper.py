import gc
import re
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def setup_pipeline(model_id: str = "openai/whisper-large"):

    '''
    Uses openai large-v3 by default
    '''
    gc.collect()                      # Python garbage collector
    torch.cuda.empty_cache()         # Clears cached memory from PyTorch
    torch.cuda.ipc_collect()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True
    )

    return pipe


def remove_filler_words(text):
    # Define filler words pattern
    fillers = r"\b(?:um|uh|ah|ok|okay|right|so|like|actually|haan)\b"

    # Remove fillers only when they are:
    # 1. Surrounded by punctuation or whitespace (not part of meaningful phrases)
    text = re.sub(rf"(?:^|\s|[.,!?]){fillers}(?=\s|[.,!?]|$)", " ", text, flags=re.IGNORECASE)

    # Remove extra spaces caused by deletions
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


if __name__ == "__main__":

    # model_id = "Oriserve/Whisper-Hindi2Hinglish-Swift" #Use finetuned whisper for hinglish
    # pipe = setup_pipeline(model_id)

    # result = pipe("src/data/class_recording_2025_03_22.mp3")

    # with open("src/output/transcription_output2.txt", "w", encoding="utf-8") as file:
    #     file.write(result["text"])

    # print("Transcription saved to transcription_output2.txt")

    with open("src/outputs/transcription_output2.txt", "r", encoding="utf-8") as file:
        original_text = file.read()

    cleaned_text = remove_filler_words(original_text)

    # Save cleaned version
    with open("src/outputs/transcription_output_cleaned.txt", "w", encoding="utf-8") as file:
        file.write(cleaned_text)

    print("Cleaned transcript saved to transcription_output_cleaned.txt")
