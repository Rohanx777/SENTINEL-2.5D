/**
 * websocket_server.h
 * ------------------
 * High-performance WebSocket server for streaming LiDAR grid data.
 * Uses uWebSockets for zero-copy, low-latency communication.
 */

#pragma once

#include <string>
#include <functional>
#include <memory>
#include <vector>

// Forward declarations (avoid pulling uWebSockets headers into every file)
namespace uWS {
    template <bool SSL, bool isServer, typename USERDATA> struct WebSocket;
    template <bool SSL> struct TemplatedApp;
}

/**
 * Callback signature for broadcasting messages to all connected clients.
 */
using BroadcastCallback = std::function<void(const std::string& message)>;

class WebSocketServer {
public:
    /**
     * Initialize WebSocket server.
     *
     * @param host      Bind address (e.g., "0.0.0.0")
     * @param port      Port number (e.g., 8000)
     * @param route     WebSocket route (e.g., "/ws/stream_map")
     */
    WebSocketServer(const std::string& host, int port, const std::string& route);
    ~WebSocketServer();

    /**
     * Start server (non-blocking).
     * Spawns a background thread running the uWebSockets event loop.
     */
    void start();

    /**
     * Stop server and close all connections.
     */
    void stop();

    /**
     * Broadcast JSON message to all connected clients.
     * Thread-safe.
     *
     * @param message   JSON string to broadcast
     */
    void broadcast(const std::string& message);

    /**
     * Get number of connected clients.
     */
    int get_client_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;

    std::string host_;
    int port_;
    std::string route_;
};
