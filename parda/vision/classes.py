"""The visual PII classes. The order is the YOLO class id and must never change after training."""
CLASSES = ["FACE", "SIGNATURE", "QR_CODE", "STAMP"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
