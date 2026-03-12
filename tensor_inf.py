import numpy as np
import cv2
import time
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

INPUT_SIZE = 550
alpha = 0.5
MEANS = np.array([103.94, 116.78, 123.68], dtype=np.float32)
STD = np.array([57.38, 57.12, 58.40], dtype=np.float32)

COLORS = np.array([
    (244,67,54),(233,30,99),(156,39,176),(103,58,183),(63,81,181),
    (33,150,243),(3,169,244),(0,188,212),(0,150,136),(76,175,80),
    (139,195,74),(205,220,57),(255,235,59),(255,193,7),(255,152,0)
], dtype=np.uint8)

class_names=['person','bicycle','car','motorcycle','airplane','bus',
'train','truck','boat','traffic light','fire hydrant','stop sign',
'parking meter','bench','bird','cat','dog','horse','sheep','cow',
'elephant','bear','zebra','giraffe','backpack','umbrella','handbag',
'tie','suitcase','frisbee','skis','snowboard','sports ball','kite',
'baseball bat','baseball glove','skateboard','surfboard','tennis racket',
'bottle','wine glass','cup','fork','knife','spoon','bowl','banana',
'apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
'donut','cake','chair','couch','potted plant','bed','dining table',
'toilet','tv','laptop','mouse','remote','keyboard','cell phone',
'microwave','oven','toaster','sink','refrigerator','book','clock',
'vase','scissors','teddy bear','hair drier','toothbrush'
]

def generate_priors():
    feature_map_sizes = [69, 35, 18, 9, 5]
    aspect_ratios = [[1, 0.5, 2]] * 5
    scales = [24, 48, 96, 192, 384]
    priors = []
    for idx, fsize in enumerate(feature_map_sizes):
        scale = scales[idx]
        for y in range(fsize):
            for x in range(fsize):
                cx, cy = (x + 0.5) / fsize, (y + 0.5) / fsize
                for ratio in aspect_ratios[idx]:
                    r = np.sqrt(ratio)
                    priors.append([cx, cy, scale/INPUT_SIZE*r, scale/INPUT_SIZE/r])
    return np.array(priors, dtype=np.float32)
def decode(loc, priors, variances=[0.1, 0.2]):
    boxes = np.empty_like(loc)
    boxes[:, :2] = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    boxes[:, 2:] = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    res = np.empty_like(boxes)
    res[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
    res[:, 2:] = res[:, :2] + boxes[:, 2:]
    return res
    
class TensorRTInference:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.priors = generate_priors()
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append((host_mem, device_mem))
            else:
                self.outputs.append((host_mem, device_mem, shape))
    def preprocess(self, img):
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img = (img.astype(np.float32) - MEANS) / STD
        img = img[:, :, ::-1].transpose(2, 0, 1)
        return np.ascontiguousarray(img[None, ...])
    def infer(self, input_data):
        host, device = self.inputs[0]
        np.copyto(host, input_data.ravel())
        cuda.memcpy_htod_async(device, host, self.stream)
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
        self.context.execute_async_v3(stream_handle=self.stream.handle)  
        results = []
        for host, device, shape in self.outputs:
            cuda.memcpy_dtoh_async(host, device, self.stream)
            results.append(host.reshape(shape)) 
        self.stream.synchronize()
        return results

    def postprocess(self, output, img_shape, score_threshold=0.3):
        loc, conf, mask_coeff, _, proto = output
        h, w = img_shape
        cur_conf = conf[0, :, 1:]
        scores = np.max(cur_conf, axis=1)
        classes = np.argmax(cur_conf, axis=1)
        keep = scores > score_threshold

        if not np.any(keep): 
            return []
        scores, classes, mask_coeff, loc = scores[keep], classes[keep], mask_coeff[0, keep], loc[0, keep]
        boxes = decode(loc, self.priors[keep])
        boxes_xywh = boxes.copy()
        boxes_xywh[:, 2:] -= boxes_xywh[:, :2]
        indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), score_threshold, 0.5) 
        if len(indices) == 0: 
            return []
        indices = np.array(indices).flatten()
        boxes, scores, classes, mask_coeff = boxes[indices], scores[indices], classes[indices], mask_coeff[indices]
        masks = proto[0] @ mask_coeff.T
        masks = 1 / (1 + np.exp(-masks)) # Sigmoid
        masks = masks.transpose(2, 0, 1)
        final_detections = []
        proto_h, proto_w = proto.shape[1:3]
        for i, m in enumerate(masks):
            x1, y1, x2, y2 = boxes[i]
            ix1, iy1 = int(max(x1*w, 0)), int(max(y1*h, 0))
            ix2, iy2 = int(min(x2*w, w)), int(min(y2*h, h))         
            if ix2 <= ix1 or iy2 <= iy1: 
               continue
            px1, py1 = int(x1 * proto_w), int(y1 * proto_h)
            px2, py2 = int(x2 * proto_w), int(y2 * proto_h)    
            m_crop = m[max(py1,0):min(py2,proto_h), max(px1,0):min(px2,proto_w)]
            if m_crop.size == 0: 
               continue
            m_res = cv2.resize(m_crop, (ix2-ix1, iy2-iy1))
            full_m = np.zeros((h, w), dtype=bool)
            full_m[iy1:iy2, ix1:ix2] = m_res > 0.5
            final_detections.append((full_m, classes[i], scores[i], (ix1, iy1, ix2, iy2)))
        return final_detections

if __name__=="__main__":
    engine_path = "/home/puja/yolact/weights/yolact_base.engine"
    model = TensorRTInference(engine_path)
    cap = cv2.VideoCapture("/home/puja/yolact/test.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        start = time.time()
        h, w = frame.shape[:2]
        inp = model.preprocess(frame)
        raw_output = model.infer(inp)
        detections = model.postprocess(raw_output, (h, w))
        if detections:
            for mask, cls, score, box in detections:
                color = COLORS[int(cls) % len(COLORS)].tolist()
                frame[mask] = frame[mask] * (1 - alpha) + np.array(color) * alpha
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_names[cls]} {score:.2f}", (x1, y1-10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        fps = 1 / (time.time() - start)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("yolact", frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
           break
    cap.release()
    cv2.destroyAllWindows()

