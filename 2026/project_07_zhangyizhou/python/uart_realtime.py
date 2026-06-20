#!/usr/bin/env python3
"""Real-time MNIST digit recognition over UART.
Usage: python uart_realtime.py COM3
       python uart_realtime.py COM3 --test  (no camera, use paint)
"""
import cv2, serial, numpy as np, sys, time, struct

MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081
QUANT_SCALE = 0.585
CAM_W, CAM_H = 640, 480
CROP_SIZE, OUT_SIZE = 320, 32

def init_camera():
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Camera found at index {idx}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            return cap
        cap.release()
    return None

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = 255 - gray
    cy, cx = gray.shape
    y0, x0 = (cy-CROP_SIZE)//2, (cx-CROP_SIZE)//2
    crop = gray[y0:y0+CROP_SIZE, x0:x0+CROP_SIZE]
    resized = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    f = resized.astype(np.float32)/255.0
    n = (f - MNIST_MEAN)/MNIST_STD
    q = np.round(n/QUANT_SCALE)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q.flatten()

def draw_overlay(frame, digit, scores):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (10, 10), (350, 280), (30, 30, 30), -1)
    frame = cv2.addWeighted(ov, 0.7, frame, 0.3, 0)
    cv2.putText(frame, f"Digit: {digit}", (20, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)
    mx = max(scores) if max(scores) > 0 else 1
    for i, s in enumerate(scores):
        sw = max(0, int(s*200/mx))
        y = 85 + i*17
        c = (0,255,0) if i==digit else (0,200,200)
        if s < 0: c = (100,100,100)
        cv2.rectangle(frame, (20, y), (20+sw, y+12), c, -1)
        cv2.putText(frame, f"{i}:{s}", (230, y+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cx = (w-CROP_SIZE)//2
    cy = (h-CROP_SIZE)//2
    cv2.rectangle(frame, (cx, cy), (cx+CROP_SIZE, cy+CROP_SIZE), (255,0,0), 2)
    return frame

def main():
    port = sys.argv[1] if len(sys.argv) > 1 else 'COM3'
    baud = 115200
    use_test = '--test' in sys.argv

    print(f"Opening {port} @ {baud} baud...")
    ser = serial.Serial(port, baud, timeout=5)
    time.sleep(1)

    if use_test:
        print("Test mode: right-click to draw, space to send")
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        drawing = False
        def mouse(event, x, y, flags, param):
            nonlocal drawing
            if event == cv2.EVENT_RBUTTONDOWN: drawing = True
            elif event == cv2.EVENT_RBUTTONUP: drawing = False
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                cv2.circle(canvas, (x,y), 8, (255,255,255), -1)
        cv2.namedWindow("Draw")
        cv2.setMouseCallback("Draw", mouse)

        while True:
            frame = canvas.copy()
            img = preprocess(frame)
            ser.write(img.tobytes()); ser.flush()
            resp = ser.readline().decode('ascii', errors='ignore').strip()
            digit = -1; scores = [0]*10
            if resp.startswith('D:'):
                try:
                    ps = resp.split()
                    digit = int(ps[0][2:])
                    scores = [int(s) for s in ' '.join(ps[1:]).replace('S:', '').split()]
                except: pass
            frame = draw_overlay(frame, digit, scores)
            cv2.imshow("Draw", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            if k == ord('c'): canvas[:] = 0  # clear
        cv2.destroyAllWindows()
        ser.close()
        return

    # Camera mode
    cap = init_camera()
    if cap is None:
        print("No camera! Use test mode: python uart_realtime.py COM3 --test")
        return

    print("Ready. q=quit, s=save")
    fps_t = time.time(); fps_n = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        img = preprocess(frame)
        ser.write(img.tobytes()); ser.flush()
        resp = ser.readline().decode('ascii', errors='ignore').strip()
        digit = -1; scores = [0]*10
        if resp.startswith('D:'):
            try:
                ps = resp.split()
                digit = int(ps[0][2:])
                scores = [int(s) for s in ' '.join(ps[1:]).replace('S:', '').split()]
            except: pass
        frame = draw_overlay(frame, digit, scores)
        fps_n += 1
        if fps_n % 30 == 0:
            el = time.time()-fps_t; fps_t=time.time()
            cv2.putText(frame, f"FPS:{30/el:.0f}", (10,470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.imshow("LeNet-5", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('s'): cv2.imwrite("capture.png", frame)
    cap.release()
    cv2.destroyAllWindows()
    ser.close()

if __name__ == '__main__':
    main()
