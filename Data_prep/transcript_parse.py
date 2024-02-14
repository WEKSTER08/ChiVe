import os

# Path to the directory containing .txt files
folder_path = "data/transcripts"

# Output file name
output_file = "data/aud_transcripts.txt"

# Open the output file in append mode
with open(output_file, "a") as outfile:
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .txt file
        if filename.endswith(".txt"):
            # Construct the full path to the file
            file_path = os.path.join(folder_path, filename)
            # Open the file and read its content
            with open(file_path, "r") as infile:
                # Read the content and write it to the output file
                content = infile.read()
                outfile.write(content)
                # Add a new line after each file's content for separation
                outfile.write("\n")

print("All .txt files have been read and merged into", output_file)
