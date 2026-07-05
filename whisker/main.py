import argparse
import logging
import sys
import cProfile
import pstats
import io


from .core.application import MainApplication


class WhiskerMainArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description="WHISKER Application")
        self.add_argument(
            "--log_level",
            default="INFO",
            choices=[logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL],
            help="Set the logging level (default: INFO)",
            type=lambda x: getattr(logging, x.upper(), logging.INFO),
        )
        self.add_argument(
            "--cli",
            action="store_true",
            help="Launch the application in headless CLI mode.",
        )
        self._add_profiling_arguments()
    
    def _add_profiling_arguments(self):
        self.add_argument(
            "--profile",
            action="store_true",
            help="Run the application under cProfile.",
        )
        self.add_argument(
            "--profile_file",
            default="whisker.prof",
            help="Output file for profiling statistics (default: whisker.prof)",
        )
        self.add_argument(
            "--print_stats",
            action="store_true",
            help="Print profiling stats to console after execution (requires --profile).",
        )
        self.add_argument(
            "--server",
            action="store_true",
            help="Start the CLI bridge server.",
        )
        self.add_argument(
            "--client",
            action="store_true",
            help="Run as a CLI bridge client, sending commands to a running server.",
        )
        self.add_argument(
            "--port",
            type=int,
            help="Port for the CLI bridge server/client.",
        )

def main(args, extra_args):
    main_application = MainApplication("MainApplication")

    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger("WHISKER")

    if args.profile:
        logger.info(f"Profiling enabled. Stats will be saved to {args.profile_file}")

    if args.cli:
        if args.server:
            from .terminal.bridge import CLIServer
            server = CLIServer(port=args.port)
            try:
                server.start()
                return 0
            except KeyboardInterrupt:
                server.shutdown()
                return 0
            except Exception as e:
                logger.critical(f"CLI Server failed: {e}", exc_info=True)
                return 1

        if args.client:
            from .terminal.bridge import CLIClient
            import json
            import shlex
            client = CLIClient(port=args.port)
            try:
                # Handle special 'batch' command for client
                if extra_args and extra_args[0] == "batch":
                    # Assume second arg is a file path
                    if len(extra_args) < 2:
                        print("Error: 'batch' command requires a file path.")
                        return 1
                    
                    file_path = extra_args[1]
                    with open(file_path, "r") as f:
                        commands = [shlex.split(line.strip()) for line in f if line.strip()]
                    
                    results = client.execute_batch(commands)
                    print(json.dumps(results, indent=2))
                    return 0
                else:
                    result = client.execute_command(extra_args)
                    print(json.dumps(result, indent=2))
                    return 0
            except Exception as e:
                logger.error(f"CLI Client failed: {e}")
                return 1

        from . import terminal
        AppEntryPoint = terminal.Application
    else:
        from . import gui
        AppEntryPoint = gui.Application

    try:
        if args.cli:
            app_instance = AppEntryPoint(extra_args)
        else:
            app_instance = AppEntryPoint(extra_args, log_level=args.log_level)
    except Exception as e:
        logger.critical(f"Failed to initialize application: {e}", exc_info=True)
        return 1

    return_code = app_instance.exec()
    main_application.shutdown()

    return return_code


def main_with_profiling(args, extra_args):
    pr = cProfile.Profile()
    try:
        pr.runctx(
            'main(args, extra_args)',
            globals(),
            {
                'args': args,
                'extra_args': extra_args
            }
        )
    except Exception as e:
        raise e
    finally:
        # Always dump stats, even if the app crashed
        pr.dump_stats(args.profile_file)
        logging.info(f"Profiling stats saved to {args.profile_file}")

        if args.print_stats:
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats('whisker')
            
            print("\n" + "="*80)
            print(f" cProfile Stats Summary (Top 30, sorted by cumulative time) ")
            print("="*80)
            print(s.getvalue())
            print(f"Full stats report saved to: {args.profile_file}")
            print("You can inspect this file visually with tools like 'snakeviz'.")
            print("="*80 + "\n")

if __name__ == "__main__":
    parser = WhiskerMainArgumentParser()
    args, extra_args = parser.parse_known_args()
    
    exit_code = 0
    try:
        if args.profile:
            exit_code = main_with_profiling(args, extra_args)
        else:
            exit_code = main(args, extra_args)
    except Exception as e:
        logging.critical(f"Application crashed: {e}", exc_info=True)
        exit_code = 1

    sys.exit(exit_code)