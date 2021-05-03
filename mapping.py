# Numbers are the number of occurances in the full dataset
TWEET_EMOTION_MAPPING = {
            "dislike": "disgust", # 898
            "anger or annoyance or hostility or fury": "anger", #568
            "admiration": "trust", #393
            "joy or happiness or elation": "joy", #308
            "disgust": "disgust", #306
            "like": "trust", #255
            "anticipation or  expectancy or interest": "anticipation", #250
            "disappointment": "sadness", #248
            "uncertainty or indecision or confusion": None, #129
            "acceptance": None, #126
            "indifference": None, #117
            "fear or apprehension or panic or terror": "fear", #91
            "vigilance": None, #68
            "hate": "disgust", #66
            "amazement": None, #65
            "surprise": "surprise", #57
            "calmness or serenity": None, #41
            "trust": "trust", #34
            "sadness or gloominess or grief or sorrow": "sadness", #31
            "BLANK": None, #1
        }

REMAN_EMOTION_MAPPING = {
        'anger':'anger',
        'anticipation':'anticipation',
        'character':None,
        'disgust':'disgust',
        'event':None,
        'fear':'fear',
        'joy':'joy',
        'other':None,
        'other-emotion':None,
        'sadness':'sadness',
        'surprise':'surprise',
        'trust':'trust'
    }

REMAN_SRL_MAPPING = {
        'cause':None, 
        'coreference':None, 
        'experiencer':'experiencer', 
        'target':'target'
    }