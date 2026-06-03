import json
import jiwer
from collections import defaultdict

def evaluate_asr_metrics(ground_truths, predictions):
    """
    ASR(음성인식) 모델의 예측 결과에 대해 WER과 CER을 정량적으로 계산합니다.
    """
    if not ground_truths or not predictions:
        return {"wer": 0.0, "cer": 0.0}
        
    #전체 리스트를 통째로 넘겨야 jiwer가 모든 단어/글자 수를 합산하여 올바른 글로벌 점수를 냅니다.
    wer_score = jiwer.wer(ground_truths, predictions)
    cer_score = jiwer.cer(ground_truths, predictions)
    
    return {
        "wer": round(wer_score, 4),
        "cer": round(cer_score, 4)
    }

if __name__ == "__main__":
    # 💡 해결책 1: 어떤 언어가 들어와도 에러가 안 나도록 defaultdict로 리스트 바구니 생성
    lang_data = defaultdict(lambda: {"gts": [], "origs": [], "advs": []})
    
    data_dir = "./data/attack_results/all_results.jsonl"
    
    try:
        with open(data_dir, "r", encoding="utf-8") as f:
            for row in f:
                if not row.strip(): 
                    continue
                result = json.loads(row)
                lang = result["lang_tag"]
                
                # 💡 해결책 2: 루프 돌 때는 계산하지 않고, 문장들을 리스트에 차곡차곡 쌓아둡니다.
                lang_data[lang]["gts"].append(result["ground_truth"])
                lang_data[lang]["origs"].append(result["clean_pred"])
                lang_data[lang]["advs"].append(result["adv_pred"])
    except FileNotFoundError:
        print(f"❌ 결과 파일({data_dir})이 없습니다. exe.py를 먼저 실행해 주세요.")
        exit()

    print("\n============= FINAL EVALUATION RESULTS =============")
    
    # 모인 데이터를 바탕으로 언어별 최종 계산
    for lang in sorted(lang_data.keys()):
        data = lang_data[lang]
        count = len(data["gts"])
        
        # 💡 한 번에 전체 문장 리스트를 넘겨 정확한 글로벌 WER/CER 산출
        gt2orig = evaluate_asr_metrics(data["gts"], data["origs"])
        gt2adv = evaluate_asr_metrics(data["gts"], data["advs"])
        orig2adv = evaluate_asr_metrics(data["origs"], data["advs"])
        
        print(f"\n🌍 Language: {lang.upper()} (Total Samples: {count})")
        print(f" 📑 GT vs Original Prediction : WER={gt2orig['wer']:.4f}, CER={gt2orig['cer']:.4f}")
        print(f" 🟢 GT vs Adversarial Pred.   : WER={gt2adv['wer']:.4f}, CER={gt2adv['cer']:.4f}")
        print(f" 🔴 Original vs Adversarial   : WER={orig2adv['wer']:.4f}, CER={orig2adv['cer']:.4f}")
    print("====================================================")