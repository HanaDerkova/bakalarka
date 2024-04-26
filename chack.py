import os

bordel = []

def check_training_logs(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == 'trainig_log':
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    if 'no_mean=True' in f.read():
                        print(f"Pattern 'no_mean=True' found in: {file_path}")
                        bordel.append(file_path)
            
folder_to_search = './outputs'  # Change this to the folder path you want to search
check_training_logs(folder_to_search)
print(len(bordel))