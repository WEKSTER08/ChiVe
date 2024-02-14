with open("data/Labels/labels.txt", "r") as lb_file:
    labels = lb_file.readlines()
    count = 0
    label_list = ["MSP-PODCAST_0003_0305.wav","MSP-PODCAST_0003_0584.wav","MSP-PODCAST_0005_0154.wav","MSP-PODCAST_0014_0009.wav","MSP-PODCAST_0014_0010.wav"]
    
    for i in range(20000):
        if labels[i].startswith("MSP-PODCAST"):
            label_vals = labels[i].split(';')
            if label_vals[0] in label_list:
                print(label_vals[0],label_vals[1])
            count +=1
    print(count)