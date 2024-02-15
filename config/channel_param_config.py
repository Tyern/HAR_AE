
cnn_channel_param = "cnn_channel_param"
pool_channel_param = "pool_channel_param"
linear_channel_param = "linear_channel_param"
model_name = "model_name"

channel_param_dict = {
    "4cnnmp_128": {
        cnn_channel_param: [
        (6, 32, 9, 0, 1),
        (32, 64, 9, 0, 1),
        (64, 128, 8, 0, 1),
        (128, 256, 6, 0, 1),
        ],
        pool_channel_param: [
            (3, 3, 0),
            (3, 3, 0),
            (3, 3, 0),
            None
        ],
        linear_channel_param: [
            128
        ],
    },

    "4cnnmp_64": {
        cnn_channel_param: [
            (6, 32, 9, 0, 1),
            (32, 64, 9, 0, 1),
            (64, 128, 8, 0, 1),
            (128, 256, 6, 0, 1),
        ],
        pool_channel_param: [
            (3, 3, 0),
            (3, 3, 0),
            (3, 3, 0),
            None
        ],
        linear_channel_param: [
            64
        ],
    },

    "4cnn_128": {
        cnn_channel_param: [
            (6, 32, 8, 0, 3),
            (32, 64, 9, 0, 3),
            (64, 128, 8, 0, 3),
            (128, 256, 7, 0, 3),
        ],
        pool_channel_param: None,
        linear_channel_param: [
            128
        ]
    },

    "4cnn_64": {
        cnn_channel_param: [
            (6, 32, 8, 0, 3),
            (32, 64, 9, 0, 3),
            (64, 128, 8, 0, 3),
            (128, 256, 7, 0, 3),
        ],
        pool_channel_param: None,
        linear_channel_param: [
            64
        ]
    },

    "poor_128": {
        cnn_channel_param: [
            (6, 64, 26, 0, 11),
            (64, 512, 22, 0, 11)
        ],
        linear_channel_param: [
            256, 128
        ]
    },

    "poor_64": {
        cnn_channel_param: [
            (6, 64, 26, 0, 11),
            (64, 512, 22, 0, 11)
        ],
        linear_channel_param: [
            256, 128, 64
        ]
    },

    "rich_128": {
        cnn_channel_param : [
            (6, 32, 8, 0, 3),
            (32, 64, 8, 0, 3)
        ],
        linear_channel_param : [
            256, 128
        ]
    },

    "rich_64": {
        cnn_channel_param : [
            (6, 32, 8, 0, 3),
            (32, 64, 8, 0, 3)
        ],
        linear_channel_param : [
            256, 64
        ]
    }
}

enc_linear_param_dict = {
    "32": [32]
}