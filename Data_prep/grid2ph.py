# import re
# with open('MSP-PODCAST_0002_0039.TextGrid','r') as f:
#   data = f.read()
# #print data #Use this to view how the code would look like after the program has opened the files
# print(data.item[1])
# txttext = ''
# for lines in data[9:]:  #informations needed begin on the 9th lines
#     # print(lines)
#     line = re.sub('\n','',lines) #as there's \n at the end of every sentence.
#     line = re.sub ('^ *','',lines) #To remove any special characters
#     linepair = line.split('=')
#     if len(linepair) == 2:
#         if linepair[0] == 'xmin':
#             xmin == linepair[1]
#         if linepair[0] == 'xmax':
#             xmax == linepair[1]
#         if linepair[0] == 'text':
#             if linepair[1].strip().startswith('"') and linepair[1].strip().endswith('"'):
#                 text = linepair[1].strip()[1:-1]
#                 txttext += text + '\n'  
# # print(txttext)
# import praatio
# from praatio import textgrid

# # Replace 'your_textgrid_file.TextGrid' with the actual path to your TextGrid file
# textgrid_path = 'MSP-PODCAST_0002_0039.TextGrid'

# # Load the TextGrid file
# tg = textgrid.openTextgrid(textgrid_path)

# # Access tiers
# for tier_name in tg.tierNameList:
#     tier = tg.tierDict[tier_name]

#     print(f"Tier Name: {tier_name}")
#     print(f"Number of intervals: {len(tier.entryList)}")

#     # Access intervals within the tier
#     for interval in tier.entryList:
#         print(f"Interval [{interval.start:.4f}, {interval.end:.4f}]: {interval.label}")
print("HI")
import sys
import textgrids

# Open the output file for writing
with open("phones_1.txt", "w") as output_file:
    # Iterate through each command-line argument
    for arg in sys.argv[1:]:
        # Try to open the file as textgrid
        try:
            grid = textgrids.TextGrid(arg)
        # Discard and try the next one
        except:
            continue

        # Assume "phones" is the name of the tier containing phone information
        for phone in grid['phones']:
            # Convert Praat to Unicode in the label
            label = phone.text.transcode()
            # Write xmin, xmax, and label to the output file
            output_file.write(f"{phone.xmin} {phone.xmax} {label}\n")
