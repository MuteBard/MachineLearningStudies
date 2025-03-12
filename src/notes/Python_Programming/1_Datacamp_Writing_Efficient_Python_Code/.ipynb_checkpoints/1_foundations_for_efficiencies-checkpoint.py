import numpy as np


# # Create a range object that goes from 0 to 5
nums = range(6)
print(type(nums))

# Convert nums to a list
nums_list = list(nums)
print(nums_list)

# Create a new list of odd numbers from 1 to 11 by unpacking a range object
nums_list2 = list(range(1,12,2))
print(nums_list2)


# # ----------------------------------------------- #
names = ['Jerry', 'Kramer', 'Elaine', 'George', 'Newman']

# Rewrite the for loop to use enumerate
indexed_names = []
for i,name in enumerate(names):
    index_name = (i,name)
    indexed_names.append(index_name) 
print(indexed_names)

# Rewrite the above for loop using list comprehension
indexed_names_comp = [(i,name) for i,name in list(indexed_names)]
print(indexed_names_comp)

# Unpack an enumerate object with a starting index of one
indexed_names_unpack = list(enumerate(names, start=1))
print(indexed_names_unpack)

# # ----------------------------------------------- #

# Use map to apply str.upper to each element in names
names_map  = map(str.upper, names)

# Print the type of the names_map
print(type(names_map))

# Unpack names_map into a list
names_uppercase = [*names_map]

# Print the list created above
print(names_uppercase)

# ----------------------------------------------- #

nums = np.array([[ 1, 2,  3,  4,  5],
 [ 6,  7,  8,  9, 10]])

# Print second row of nums
print(nums[1])

# Print all elements of nums that are greater than six
print(nums[nums > 6])

# Double every element of nums
nums_dbl = nums * 2
print(nums_dbl)

# Replace the third column of nums
nums[:,2] = nums[:,2] + 1
print(nums)

# --------

arrival_times = [*range(10,60,10)]
# arrival_times = list(range(10, 60, 10))
print(arrival_times)

arrival_times_np = np.array(arrival_times)
new_times = arrival_times_np - 3

print(new_times)


# Use list comprehension and enumerate to pair guests to new times
guest_arrivals = [(names[i],time) for i,time in enumerate(new_times)]



def welcome_guest(ga):
    return f"""Welcome to Festivus {ga[0]}... You're {ga[1]} min late"""


# Map the welcome_guest function to each (guest,time) pair
welcome_map = map(welcome_guest, guest_arrivals)

guest_welcomes = [*welcome_map]
print(*guest_welcomes, sep='\n')

# Welcome to Festivus Jerry... You're 7 min late.
# Welcome to Festivus Kramer... You're 17 min late.
# Welcome to Festivus Elaine... You're 27 min late.
# Welcome to Festivus George... You're 37 min late.
# Welcome to Festivus Newman... You're 47 min late.