import os 

model_path = {
	'ffhq': {
		'e4e': 'pretrained_models/e4e/e4e_ffhq_encode.pt',
        'tabnet': 'pretrained_models/TabNet/ffhq_gender.pt',
        'xgboost': None,
	},
    
    'mnist': {
        'e4e': '',
        'tabnet': '',
        'xgboost': None,
    },
 
	'afhq': {
		'e4e': 'pretrained_models/e4e/e4e_afhq_2nd.pt',
        'tabnet': 'pretrained_models/TabNet/afhq_2nd.pt',
        'xgboost': None,
	},
	'leaf': {
        'e4e': 'pretrained_models/e4e/e4e_leaf_encode.pt',
        'tabnet': 'pretrained_models/TabNet/leaf_binary.pt',
        'xgboost': None,
    }
}

sample_path = {
    'ffhq': {
        'male': [
            ""
        ],
        'female': [

        ],
    },
    'afhq': {
        'cat': [
            "datasets/afhq/test/cat/flickr_cat_000008.jpg"
        ],
        'dog': [
            "datasets/afhq/test/dog/flickr_dog_000094.jpg"
        ],
    },
    'leaf': {
        'healthy': [
            "datasets/Leaf/status/test/healthy/Apple___healthy_image (12).JPG",
            "datasets/Leaf/status/test/healthy/Apple___healthy_image (14).JPG",
            "datasets/Leaf/status/test/healthy/Apple___healthy_image (19).JPG",
            "datasets/Leaf/status/test/healthy/Apple___healthy_image (22).JPG",
            "datasets/Leaf/status/test/healthy/Apple___healthy_image (38).JPG",
            "datasets/Leaf/status/test/healthy/Apple___healthy_image (54).JPG",
            "datasets/Leaf/status/test/healthy/Apple___healthy_image (68).JPG",
            "datasets/Leaf/status/test/healthy/Apple___healthy_image (75).JPG",
            ],
        'sick': [
            "datasets/Leaf/status/test/sick/Apple___Apple_scab_image (5).JPG",
        ],
    } 
}
