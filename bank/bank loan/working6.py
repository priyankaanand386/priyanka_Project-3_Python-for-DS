import bcrypt

# Hashing the password
password = "password123"
hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Comparing the password
entered_password = "password123"
if bcrypt.checkpw(entered_password.encode('utf-8'), hashed_password):
    print("Password matches!")
else:
    print("Incorrect password!")
