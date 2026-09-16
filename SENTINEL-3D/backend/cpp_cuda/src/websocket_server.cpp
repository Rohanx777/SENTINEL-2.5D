/**
 * websocket_server.cpp
 * --------------------
 * uWebSockets-based high-performance WebSocket server implementation.
 */

#include "websocket_server.h"
#include <App.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <set>

// Per-WebSocket user data
struct PerSocketData {
    int client_id;
};

// Internal implementation (pimpl idiom to hide uWebSockets from header)
struct WebSocketServer::Impl {
    std::unique_ptr<std::thread> server_thread;
    std::atomic<bool> running{false};
    std::atomic<int> client_count{0};

    uWS::App* app = nullptr;
    us_listen_socket_t* listen_socket = nullptr;

    std::mutex clients_mutex;
    std::set<uWS::WebSocket<false, true, PerSocketData>*> clients;

    std::string host;
    int port;
    std::string route;

    ~Impl() {
        if (listen_socket) {
            us_listen_socket_close(0, listen_socket);
        }
        delete app;
    }
};

WebSocketServer::WebSocketServer(const std::string& host, int port, const std::string& route)
    : pimpl_(std::make_unique<Impl>())
    , host_(host)
    , port_(port)
    , route_(route)
{
    pimpl_->host = host;
    pimpl_->port = port;
    pimpl_->route = route;
}

WebSocketServer::~WebSocketServer() {
    stop();
}

void WebSocketServer::start() {
    if (pimpl_->running.load()) {
        std::cerr << "[WebSocketServer] Already running" << std::endl;
        return;
    }

    pimpl_->running.store(true);

    pimpl_->server_thread = std::make_unique<std::thread>([this]() {
        pimpl_->app = new uWS::App();

        pimpl_->app->ws<PerSocketData>(
            pimpl_->route,
            {
                .compression = uWS::DISABLED,
                .maxPayloadLength = 16 * 1024 * 1024,
                .idleTimeout = 120,
                .maxBackpressure = 1 * 1024 * 1024,

                .open = [this](auto* ws) {
                    std::lock_guard<std::mutex> lock(pimpl_->clients_mutex);
                    pimpl_->clients.insert(ws);
                    pimpl_->client_count.fetch_add(1);

                    auto* data = ws->getUserData();
                    data->client_id = pimpl_->client_count.load();

                    std::cout << "[+] Client connected (total: "
                              << pimpl_->clients.size() << ")" << std::endl;
                },

                .message = [](auto* ws, std::string_view message, uWS::OpCode opCode) {
                    // We don't expect client → server messages in this use case
                    // If needed, handle incoming control messages here
                },

                .close = [this](auto* ws, int code, std::string_view message) {
                    std::lock_guard<std::mutex> lock(pimpl_->clients_mutex);
                    pimpl_->clients.erase(ws);
                    pimpl_->client_count.fetch_sub(1);

                    std::cout << "[-] Client disconnected (total: "
                              << pimpl_->clients.size() << ")" << std::endl;
                }
            }
        )
        .listen(pimpl_->host, pimpl_->port, [this](auto* listen_socket) {
            if (listen_socket) {
                pimpl_->listen_socket = listen_socket;
                std::cout << "[WebSocketServer] Listening on ws://"
                          << pimpl_->host << ":" << pimpl_->port << pimpl_->route
                          << std::endl;
            } else {
                std::cerr << "[WebSocketServer] Failed to bind to "
                          << pimpl_->host << ":" << pimpl_->port << std::endl;
                pimpl_->running.store(false);
            }
        })
        .run();

        std::cout << "[WebSocketServer] Event loop exited" << std::endl;
    });
}

void WebSocketServer::stop() {
    if (!pimpl_->running.load()) return;

    pimpl_->running.store(false);

    // Close listen socket to stop accepting new connections
    if (pimpl_->listen_socket) {
        us_listen_socket_close(0, pimpl_->listen_socket);
        pimpl_->listen_socket = nullptr;
    }

    // Wait for server thread to finish
    if (pimpl_->server_thread && pimpl_->server_thread->joinable()) {
        pimpl_->server_thread->join();
    }

    std::cout << "[WebSocketServer] Stopped" << std::endl;
}

void WebSocketServer::broadcast(const std::string& message) {
    if (!pimpl_->running.load()) return;

    std::lock_guard<std::mutex> lock(pimpl_->clients_mutex);

    for (auto* ws : pimpl_->clients) {
        // Send with compression disabled for lowest latency
        ws->send(message, uWS::OpCode::TEXT);
    }
}

int WebSocketServer::get_client_count() const {
    return pimpl_->client_count.load();
}
