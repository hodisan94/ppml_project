# import flwr as flower
#
# NUM_CLIENTS = 5
#
# strategy = flower.server.strategy.FedAvg(
#     fraction_fit=1.0,
#     min_fit_clients=NUM_CLIENTS,
#     min_eval_clients=NUM_CLIENTS,
#     min_available_clients=NUM_CLIENTS,
# )
#
# flower.server.start_server(
#     server_address="127.0.0.1:8086",
#     config=flower.server.ServerConfig(num_rounds=5),
#     strategy=strategy,
# )
import flwr as flower

NUM_CLIENTS = 5

# Define strategy with correct parameter names for newer Flower versions
strategy = flower.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,  # Changed from fraction_eval
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,  # Changed from min_eval_clients
    min_available_clients=NUM_CLIENTS,
)

print("[SERVER] Starting Flower server...")
print(f"[SERVER] Waiting for {NUM_CLIENTS} clients to connect...")

# Start server
flower.server.start_server(
    server_address="127.0.0.1:8086",
    config=flower.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)

print("[SERVER] Server finished training rounds.")