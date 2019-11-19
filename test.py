from darkflow.net.build import TFNet
import cv2

threshold = 0.03
cfg_path = "cfg/darkflow-ogar-loadweights.cfg"
test_img_path = "data/ogars_in_flight/images/Ogar00131.png"

darkflow_flags = {
    "model": cfg_path,
    "threshold": threshold,
    "gpu": 0.9,
    "pbLoad": "built_graph/darkflow-ogar-loadweights.pb",
    "metaLoad": "built_graph/darkflow-ogar-loadweights.meta",
    "load": "bin/tiny-yolo-voc.weights",
}

tfnet = TFNet(darkflow_flags)

imgcv = cv2.imread(test_img_path)
result = tfnet.return_predict(imgcv)

#print("result:", result)
if len(result) == 0:
    print("no objects found")
else:
    best_box = result[0]
    max_conf = result[0]["confidence"]
    
    for box in result:
        if box["confidence"] > max_conf:
            best_box = box
            max_conf = box["confidence"]

    print(best_box)

#for best_box in result:
corners = [(best_box["topleft"]["x"], best_box["topleft"]["y"]),(best_box["bottomright"]["x"], best_box["bottomright"]["y"])]
cv2.rectangle(imgcv, corners[1], corners[0], (0, 0, 255), 2)
print("writing to ./test_img_show.png")
cv2.imwrite("test_img_show.png", imgcv)

