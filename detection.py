import jetson_inference
import jetson_utils

net = jetson_inference.detectNet(
    model="/home/nvidia/jetson-inference/build/aarch64/bin/networks/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff",
    labels="/home/nvidia/jetson-inference/build/aarch64/bin/networks/SSD-Mobilenet-v2/ssd_coco_labels.txt",
    input_blob="Input",
    output_cvg="NMS",
    output_bbox="NMS_1",
    threshold=0.2
)

img_path = "/home/nvidia/jetson-inference/data/images/banana_0.jpg"
img = jetson_utils.loadImage(img_path)

if img is None:
    print(f"图片加载失败: {img_path}")
    exit()

detections = net.Detect(img)

print("=" * 60)
print(f"检测图片: {img_path}")
print(f"检测目标数: {len(detections)}")
print("=" * 60)

for idx, det in enumerate(detections):
    print(f"\n--- 目标 {idx+1} ---")
    print(f"ClassID:    {det.ClassID}")
    print(f"Confidence: {det.Confidence:.6f}")
    print(f"Left:       {det.Left:.3f}")
    print(f"Top:        {det.Top:.3f}")
    print(f"Right:      {det.Right:.3f}")
    print(f"Bottom:     {det.Bottom:.3f}")
    print(f"Width:      {det.Width:.3f}")
    print(f"Height:     {det.Height:.3f}")
    print(f"Area:       {det.Area:.3f}")
    print(f"Center:     ({det.Center[0]:.3f}, {det.Center[1]:.3f})")

output_path = "result_banana_0.jpg"
jetson_utils.saveImage(output_path, img)
print(f"\n结果图已保存: {output_path}")
