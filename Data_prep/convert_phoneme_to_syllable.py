1. #converting phonetics to syllables and generating an output
import scipy.io
# import python3
# import syllabifier
from syllabify import syllabifier
import os
# from python.syllabify import syllabifier

folder_path = 'data/phone'  # Replace with the actual folder path

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Iterate over each file in the folder
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)  # Create the full file path

    # Check if the current item is a file (not a subfolder)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            # Read the contents of the file
            data = file.read().replace('\n', ' ')
            data = data.replace('\t', ' ')
            data_arr = data.split(' ')

            final = ''
            start = []
            end = []
            ph = []

            for i in range(len(data_arr)):
                if i % 3 == 2 and data_arr[i] != "sil" and data_arr[i] != "SPN":
                    ph.append(data_arr[i])
                    final += data_arr[i].upper() + ' '
                if i % 3 == 0 and i + 2 < len(data_arr) and data_arr[i + 2] != "sil":
                    start.append(data_arr[i])
                if i % 3 == 1 and i + 1 < len(data_arr) and data_arr[i + 1] != "sil":
                    end.append(data_arr[i])

            # Remove leading/trailing whitespaces and skip if the final string is empty
            if final.strip():
                # Remove the last character (space) from the final string
                final = final[:-1]

                # Exclude the "SPN" phoneme from the final string
                final = ' '.join([phoneme for phoneme in final.split() if phoneme != "SPN"])

                # Exclude any phoneme with the format ".1"
                final = ' '.join([phoneme for phoneme in final.split() if not phoneme.endswith('.1')])

                try:
                    syllables = syllabifier.syllabify(syllabifier.English, final)
                    ret = []
                    for syl in syllables:
                        stress, onset, nucleus, coda = syl
                        if stress is not None and len(nucleus) != 0:
                            nucleus[0] += str(stress)
                        ret.append(" ".join(onset + nucleus + coda))
                    output = " . ".join(ret)
                    arr = output.split(".")

                    final_st = []
                    final_end = []
                    syl = []

                    c = 0
                    for e in arr:
                        phoneme = e.split(" ")
                        while "" in phoneme:
                            phoneme.remove("")

                        if c < len(start):
                            final_st.append(start[c])
                        if c + len(phoneme) - 1 < len(end):
                            final_end.append(end[c + len(phoneme) - 1])
                        syl.append(phoneme)
                        c += len(phoneme)

                    st = []
                    for s in syl:
                        ss = ""
                        for ph in s:
                            temp = ph.lower()
                            if len(temp) == 3:
                                temp = temp[:-1]
                            ss += temp + " "
                        if len(ss):
                            ss = ss[:-1]
                            st.append(ss)
                    
                    h_output = "data/syllable" + file_name  # Replace with the desired output file path
#                     print(h_output)
#                     with open(h_output, "w") as output_file:
                    for i in range(min(len(syllables), len(st))):
                            stress = syllables[i][0]
                            onset = syllables[i][1]
                            nucleus = syllables[i][2]
                            coda = syllables[i][3]
                            onset = [ph for ph in onset if not any(char.isdigit() for char in ph)]
                            nucleus = [ph for ph in nucleus if not any(char.isdigit() for char in ph)]
                            coda = [ph for ph in coda if not any(char.isdigit() for char in ph)]

                            stress = '1' if stress else '0'
                            transcript = st[i]

                            output_line = f"{stress:<2}{final_st[i]:>7}{final_end[i]:>7} {transcript:>8}"
                            # output_file.write(output_line + "\n")
                            print(output_line)

                except ValueError as e:
            
                    print(f"Error: {e}")
# This code snippet demonstrates the conversion of phonetic representations into syllables and generates an output based on the processed data. It operates on a folder, FA_result, containing files representing phonetic transcriptions. The code reads each file, extracts phonetic information, and converts it into syllables using the syllabifier module. The extracted phonetic information is processed to remove unnecessary elements such as silences and special tokens ('SPN'). The remaining phonemes are then converted into syllables based on stress patterns. The resulting syllables are further split into onset, nucleus, and coda components. The code generates an output file with the converted syllables and associated information. Each line of the output represents a syllable and includes the syllable stress indicator, start and end times, and the corresponding transcript. The output file is currently not written, but the lines are printed to the console. In case of any errors during the conversion process, the code catches ValueError exceptions and prints an error message. This ensures that the code can handle unexpected data or issues that may arise during the conversion.
