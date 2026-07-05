import argparse
import logging
from pathlib import Path
import sys
import threading
import time

from .server import BaseServer
from .client import BaseClient, DEFAULT_MAX_RETRIES, DEFAULT_RETRY_DELAY_SEC
from .protocol import (
    ShutdownRequest,
    EchoRequest,
    TaskInitiatedResponse,
    QueryTaskStatusRequest,
    QueryTaskStatusResponse,
    TaskStatus
)
    
def run_server(server: BaseServer):
    """Run the server and handle keyboard interrupt for clean shutdown."""
    try:
        server.start()
    except KeyboardInterrupt:
        pass # Allow the main thread to handle the overall shutdown

def run_client_demo(args: argparse.Namespace):
    """Initializes client and sends an EchoRequest, polling if an async task is started."""
    logger = logging.getLogger("demo.client_runner")
    client = BaseClient(
        host=args.host,
        port=args.port,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    try:
        req = EchoRequest(message=args.message, delay=args.echo_delay)
        logger.info("Sending request: %s", type(req).__name__)
        response = client.send_request(req)

        # Polling Logic for Async Tasks
        if isinstance(response, TaskInitiatedResponse):
            task_id = response.task_id
            logger.info(f"Task initiated, ID: {task_id}. Entering polling loop...")
            print("--- Task Initiated ---")
            print(response)
            print("----------------------")

            POLL_INTERVAL_SEC = 0.5
            total_time = 0.0

            while True:
                time.sleep(POLL_INTERVAL_SEC)
                total_time += POLL_INTERVAL_SEC
                
                # Use a new connection for the query
                query_req = QueryTaskStatusRequest(task_id=task_id)
                query_response = client.send_request(query_req)

                if not isinstance(query_response, QueryTaskStatusResponse):
                    logger.error("Received unexpected response during task query: %s", type(query_response).__name__)
                    raise ConnectionError("Invalid response during task status query.")
                
                status = query_response.status
                print(f"[Time: {total_time:.1f}s] Task Status: {status}. Message: {query_response.message}")

                if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.UNKNOWN):
                    # We've done it! The task has resolved.
                    logger.info(f"Task {task_id} resolved with status: {status}")
                    # In a real system, we'd query one last time to get the final result payload.
                    # For this demo, we'll just show the final status message.
                    print("\n--- Final Task Result ---")
                    print(query_response)
                    print("-------------------------")
                    break
        else:
            # Print the useful representation of the synchronous response object
            print("--- Response Received (Synchronous) ---")
            print(response)
            print("---------------------------------------")
        
        logging.info("Sending ShutdownRequest to server.")
        client.send_request(ShutdownRequest())

        sys.exit(0)

    except ConnectionError as ce:
        logger.error("Connection error occurred: %s", ce)
        print("Connection error occurred:", ce, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        print("Unhandled error:", e, file=sys.stderr)
        sys.exit(1)


def main(ServerClass: type[BaseServer] = BaseServer):
    """Main function for the combined client/server demo."""
    parser = argparse.ArgumentParser(
        prog="bridge-demo",
        description="Run a DemoServer and an EchoClient request in a combined script."
    )
    # Server arguments (optional, defaults)
    parser.add_argument("--host", "-H", default=None, help=f"Server host (will use the class's default if not set)")
    parser.add_argument("--port", "-p", type=int, default=None, help="Server port")
    parser.add_argument("--no-launch-server", action="store_true", help="Do not launch the server.")

    # Client-specific arguments
    parser.add_argument("--message", "-m", default="Hello, Server!", help="Message to send in EchoRequest")
    parser.add_argument("--echo-delay", type=float, default=0.0, help="Delay in seconds before server echoes back the message")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help=f"Max connection retries (default: {DEFAULT_MAX_RETRIES})")
    parser.add_argument("--retry-delay", type=int, default=DEFAULT_RETRY_DELAY_SEC, help=f"Seconds between retries (default: {DEFAULT_RETRY_DELAY_SEC})")
    parser.add_argument("--no-launch-client", action="store_true", help="Do not launch the client.")

    # Logging argument
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Logging level")

    args = parser.parse_args()

    # Set up logging globally
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("demo.main")

    server = None
    server_thread = None
    try:
        if args.no_launch_server:
            logging.info("No server requested; will connect client to pre-existing server")
        else:
            server = ServerClass(host=args.host, port=args.port)
            server_thread = threading.Thread(target=run_server, args=(server,), daemon=True)            
            logger.info(f"Started Server of type {ServerClass.__name__} at {server.host}:{server.port}")
            server_thread.start()
            
            # Give the server a moment to bind and start listening
            time.sleep(0.5) 
        
        if args.no_launch_client:
            logger.info("No client requested; waiting for user to press Enter to exit.")
            try:
                input("Press Enter to exit...\n")
            except KeyboardInterrupt:
                logger.info("Interrupted while waiting for Enter.")
        else:
            logger.info("Running client demo...")
            args.host = server.host if server else args.host
            args.port = server.port if server else args.port if args.port else ServerClass.default_port()
            run_client_demo(args)

    except Exception as e:
        logger.critical("Critical error in main demo run: %s", e, exc_info=True)
    finally:
        if server:
            logger.info("Shutting down server.")
            server.shutdown()
        if server_thread:
            server_thread.join(timeout=2.0)
            if server_thread.is_alive():
                logging.warning("Server thread didn't join within timeout.")
        logger.info("Demo finished.")


if __name__ == "__main__":
    main()
