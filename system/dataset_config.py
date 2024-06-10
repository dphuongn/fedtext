def get_num_labels(dataset_name):
    num_labels = {
        "ag": 4,

        "sogou_news": 5,
        
        "dbpedia": 2,

        "cola": 2,

        "dbpedia_14": 14,
        
    }
    
    
    return num_labels[dataset_name]