class Config:
    # serverIP = 'localhost'
    # serverIP = '192.168.31.157'  # Lab407-1
    # serverIP = '192.168.31.59'   # Lab428-3
    serverIP = '192.168.31.94'   # Lab428-1
    serverPort = 2000

    cameraPaths = {
        "rgb": "background",
        "depth": "depth",
        "is": "instance_segmentation",
        "ss": "semantic_segmentation"
    }

    imgQueueTimeout = 5