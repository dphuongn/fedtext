def get_num_labels(dataset_name):
    num_labels = {
        "ag_news": 4,

        "sogou_news": 5,

        "cola": 2,

        "dbpedia_14": 14,

        "yelp_preview_full": 5,

        "sst2": 2,
        
    }
    
    
    return num_labels[dataset_name]