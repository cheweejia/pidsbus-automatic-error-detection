# Label map for reference.
# Erroneous classes are EDSLit, EDSEmpty, EDSHalfLit and EDSHalfEmpty (index 3 to 6).
# EDSRefresh and EDSGlare are assumed to be non-erroneous, since the causes are not within 
# our scope of research (environmental and equipment factors).
categories = ['SDFront', 'DDFront', 'EDS', 'EDSLit', 'EDSEmpty', 'EDSHalfLit', 'EDSHalfEmpty', 'EDSRefresh', 'EDSGlare']

def detect_error(detections):
    class_ids = detections['detection_classes']
    for i in range(class_ids.size):
        class_index = class_ids[i]
        if class_index > 2 and class_index < 7:
            return True
    return False