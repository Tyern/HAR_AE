import numpy as np

def limit_filter_data_by_class(data, label, limit_number=-1):
    # limit number is the maximum number of instance to filter each class
    print("limit_and_set_train_data", "limit_number=", limit_number)
    assert len(data) == len(label)

    if limit_number == -1:
        return (data, label, np.arange(len(label)))
    else:
        choice_limited_list = []

        label_set = set(label)
        for i in label_set:
            one_class_idx = np.where(label == i)[0]
            choice_idx_list = np.random.choice(
                one_class_idx, 
                min(limit_number, len(one_class_idx)), 
                replace=False)
            
            choice_limited_list.extend(choice_idx_list)

        limited_train_data = data[choice_limited_list]
        limited_train_label = label[choice_limited_list]

        print("limit_filter_data_by_class: ", np.unique(limited_train_label, return_counts=True))
    
        return (limited_train_data, limited_train_label, np.array(choice_limited_list))