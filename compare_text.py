import torch
from sentence_transformers import SentenceTransformer
docs = ["환기가 부족하면 강의실 CO2 농도가 1000ppm을 넘어 집중력이 떨어진다.",
        "에어컨 설정 온도는 하계 26도, 동계 20도가 권장된다.",
        "조도가 300lux 미만이면 필기 작업 시 눈의 피로가 커진다.",
        "Jetson Orin Nano는 67 TOPS의 AI 성능을 제공한다."]
names = ["CO2", "에어컨", "조도", "Jetson"]
queries = ["수업 중에 졸리고 머리가 맑지 않아요", "여름에 냉방 몇 도로 맞춰야 하나요?"]
for path in ["google/embeddinggemma-2", "models/eg2-classroom"]:
    m = SentenceTransformer(path, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16},
                            config_kwargs={"vision_config": None, "audio_config": None})
    d = m.encode(docs, prompt_name="Document", normalize_embeddings=True)
    q = m.encode(queries, prompt_name="SearchQuery", normalize_embeddings=True)
    print("\n==", path)
    for qi, row in zip(queries, q @ d.T):
        print(" ", qi, " ".join(f"{n}={s:.3f}" for n, s in zip(names, row)))
    del m; torch.cuda.empty_cache()
