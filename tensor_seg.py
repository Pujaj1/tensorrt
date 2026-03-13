import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import depthai as dai

ENGINE_PATH = "/home/puja/depthai-core/examples/python/DetectionNetwork/v5.engine"
CLASSES_PATH = "/home/puja/yolov5/runs/train-seg/exp2/weights/coco.names"

INPUT_WIDTH, INPUT_HEIGHT = 640, 640
CONF_THRESHOLD, NMS_THRESHOLD = 0.5, 0.45
MASK_THRESHOLD, ALPHA = 0.25, 0.4
NUM_CLASSES, NUM_MASKS = 80, 32

with open(CLASSES_PATH, "r") as f:
    class_names = [c.strip() for c in f.readlines()]

logger = trt.Logger(trt.Logger.WARNING)

def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()
stream = cuda.Stream()
outputs = []
bindings = []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    shape = engine.get_tensor_shape(name)
    size = trt.volume(shape)

    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)

    bindings.append(int(device_mem))

    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        input_data = {'host': host_mem, 'device': device_mem}
    else:
        outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})

for i in range(engine.num_io_tensors):
    context.set_tensor_address(engine.get_tensor_name(i), bindings[i])

def trt_inference(frame):
    blob = cv2.dnn.blobFromImage(
        frame, 1/255.0,
        (INPUT_WIDTH, INPUT_HEIGHT),
        swapRB=True,
        crop=False
    ).flatten()

    np.copyto(input_data['host'], blob)
    cuda.memcpy_htod_async(input_data['device'], input_data['host'], stream)

    context.execute_async_v3(stream_handle=stream.handle)

    results = []
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        results.append(out['host'].reshape(out['shape']))

    stream.synchronize()
    return results

def post_process(frame, trt_results):
    detections = trt_results[0] if trt_results[0].shape[1] > 32 else trt_results[1]
    protos = trt_results[1] if trt_results[1].shape[1] == 32 else trt_results[0]
    detections = np.squeeze(detections)
    img_h, img_w = frame.shape[:2]
    proto_h, proto_w = protos.shape[2], protos.shape[3]
    x_factor, y_factor = img_w / INPUT_WIDTH, img_h / INPUT_HEIGHT

    obj_conf = detections[:, 4]
    class_scores = detections[:, 5:5+NUM_CLASSES]
    class_ids_all = np.argmax(class_scores, axis=1)
    confidences_all = obj_conf * detections[range(len(detections)), 5 + class_ids_all]

    mask = confidences_all > CONF_THRESHOLD
    valid_detections = detections[mask]
    valid_confidences = confidences_all[mask]
    valid_class_ids = class_ids_all[mask]

    if len(valid_detections) == 0:
        return frame

    boxes, confidences, class_ids, mask_coeffs = [], [], [], []
    for i in range(len(valid_detections)):
        det = valid_detections[i]
        cx, cy, w, h = det[:4]
        left = int((cx - w / 2) * x_factor)
        top  = int((cy - h / 2) * y_factor)
        width  = int(w * x_factor)
        height = int(h * y_factor)
        
        boxes.append([left, top, width, height])
        confidences.append(float(valid_confidences[i]))
        class_ids.append(valid_class_ids[i])
        mask_coeffs.append(det[5+NUM_CLASSES:])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    if len(indices) == 0:
        return frame
        
    proto_flat = protos[0].reshape(NUM_MASKS, -1) 

    for i in indices.flatten():
        l, t, w, h = boxes[i]
        l, t = max(0, l), max(0, t)
        r, b = min(img_w, l+w), min(img_h, t+h)

        if r <= l or b <= t: 
           continue
        
        mc = mask_coeffs[i]
        m = mc @ proto_flat
        m = 1 / (1 + np.exp(-m))
        m = m.reshape(proto_h, proto_w)
        
        lp, tp = int(l * proto_w / img_w), int(t * proto_h / img_h)
        rp, bp = int(r * proto_w / img_w), int(b * proto_h / img_h)
        
        mask_crop = m[tp:bp, lp:rp]
        if mask_crop.size == 0: 
           continue
        
        mask_resized = cv2.resize(mask_crop, (r-l, b-t)) > MASK_THRESHOLD
        color = np.random.randint(0,255,(3,),dtype=np.uint8)
        
        roi = frame[t:b, l:r]
        roi[mask_resized] = (roi[mask_resized] * (1-ALPHA) + color * ALPHA)

        cv2.rectangle(frame,(l,t),(r,b),color.tolist(),2)
        cv2.putText(frame, class_names[class_ids[i]], (l,t-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)

    return frame

with dai.Pipeline() as pipeline:

    cam = pipeline.create(dai.node.Camera).build()

    cam_out = cam.requestOutput(
        size=(INPUT_WIDTH, INPUT_HEIGHT),
        type=dai.ImgFrame.Type.BGR888i
    )
    qRgb = cam_out.createOutputQueue(maxSize=1, blocking=False)
    pipeline.start()
    while pipeline.isRunning():

        inRgb = qRgb.get()   
        
        frame = inRgb.getCvFrame()
        start = time.time()
        results = trt_inference(frame)
        frame = post_process(frame, results)
        end = time.time()
        fps = 1/(end-start)
        cv2.putText(frame,f"FPS: {fps:.1f}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)
        cv2.imshow("frames",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
