CONFIG = {
    # Paths (relative to multicue-df/ root)
    "dataset_root": "FaceForensics++_C23",
    "frames_dir": "frames",
    "checkpoints_dir": "checkpoints",
    "logs_dir": "logs",
    "results_dir": "results",
    "plots_dir": "plots",

    # Dataset classes
    "real_folder": "original",
    "fake_folders": [
        "Deepfakes", "Face2Face", "FaceShifter",
        "FaceSwap", "NeuralTextures", "DeepFakeDetection"
    ],

    # Sampling
    "videos_per_class": 200,
    "frames_per_video": 10,
    "seed": 42,

    # Image
    "image_size": 224,

    # Training
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "dropout_rates": [0.3, 0.5],

    # Split
    "train_split": 0.70,
    "val_split": 0.15,
    "test_split": 0.15,

    # Model feature dims
    "stream1_dim": 512,
    "stream2_dim": 256,
    "stream3_dim": 128,
    "fused_dim": 896,

    # Device
    "device": "cuda"  # forced CUDA
}
