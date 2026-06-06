import sys
from pathlib import Path
from PIL import Image

# 프로젝트 경로 설정
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from renderer import taichi_renderer as tr

def main():
    print("1. 설정(Config) 불러오는 중...")
    cfg = tr.load_config()
    
    print("2. GPU 및 우주 배경 초기화 중...")
    tr.setup_renderer(cfg)
    
    # 썸네일 해상도 설정 (정사각형 1024x1024)
    res = 1024
    
    print(f"3. 🚀 [방식 1] render_pipe_a 실행 중... ({res}x{res})")
    # 여기가 바로 유저님이 수정한 Fix 1번 코드가 돌아가는 곳입니다!
    hdr = tr.render_pipe_a_image(cfg, res, lod_enabled=True)
    
    print("4. 이미지 톤매핑 및 저장 중...")
    # render.yaml에 있는 thumb 노출값과 감마값을 사용
    exposure = float(cfg["thumb"].get("exposure", 1.0))
    gamma = float(cfg["thumb"]["gamma"])
    img = tr.tonemap(hdr, exposure, gamma)
    
    out_path = _ROOT / "scripts" / "pipe_a_fixed.png"
    Image.fromarray(img, mode="RGB").save(str(out_path))
    print(f"✅ 완료! 결과물이 저장되었습니다: {out_path}")

if __name__ == "__main__":
    main()