olora_requests = [[(2,"str2"),(3,"str3"),(4,"str4")],[(1,"str1"),(2,"str2"),(3,"str3")]]
loras_map = {
            lora_request[0]: lora_request[1]
            for olora_request in olora_requests if olora_request
            for lora_request in olora_request
        }
print(loras_map)
a = set(loras_map)
print(set(loras_map))