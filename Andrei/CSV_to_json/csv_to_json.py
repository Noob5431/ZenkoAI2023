import csv
import json

# stand = {
#     "place_number" :
#     "booking_weezpay_account":
#     "name":
#     "drinks":
#     "foods":
# }

with open("file.csv") as csv_file:
    stall_list = []
    csv_reader = csv.reader(csv_file,delimiter=';')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            # print(f'column names are {", ".join(row)}')
            line_count += 1
        else:
            # print(f'\t{row[0]}  {row[1]}  {row[2]} {row[3]}.')
            new_stall = dict(place_number = row[0],\
                             booking_weezpay_account = row[1],\
                             name = row[2],\
                             drinks = row[3],\
                             foods = row[4])
            stall_list.append(new_stall)
            line_count += 1
        print(f'Processed {line_count} lines.')
        result = json.dumps(stall_list)
        output = open("file.json", 'w')
        output.write(result)
        
