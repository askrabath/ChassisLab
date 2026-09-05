import argparse
import os
import sys
from dotenv import load_dotenv


def main():
    p=argparse.ArgumentParser(description='VEX-inspired MuJoCo chassis experiments')
    sub=p.add_subparsers(dest='command',required=True)
    parser=sub.add_parser('assembly',help='Build and settle a fixed V5-style chassis with rails, axles and fasteners')
    parser.add_argument('--output',default='assemblies/standard_v5')
    parser.add_argument('--viewer',action='store_true',help='Open live MuJoCo viewer (use mjpython on macOS)')
    parser.add_argument('--no-render',action='store_true')
    for command in ['demo','experiment']:
        parser=sub.add_parser(command)
        if command=='demo': parser.add_argument('--offline',action='store_true',required=True)
        else:
            parser.add_argument('--ai',action='store_true',required=True)
            parser.add_argument('--generations',type=int,choices=[2],default=2,help='Exactly an initial proposal and one revision')
        parser.add_argument('--model',default=os.environ.get('CHASSIS_LAB_MODEL','gpt-5.6-sol'))
        parser.add_argument('--output',default='runs')
        parser.add_argument('--no-render',action='store_true')
    parser=sub.add_parser('replay');parser.add_argument('run_directory')
    parser.add_argument('--candidate',type=int,choices=[0,1],default=0)
    parser.add_argument('--seed',type=int,default=101);parser.add_argument('--split',choices=['development','final'],default='final')
    parser.add_argument('--headless',action='store_true')
    from .override.cli import add_commands,dispatch
    add_commands(sub)
    args=p.parse_args()
    if args.command in ['field','seed-robot','mechanism-benchmark','search','resume','mechanism-replay','compare','cad-view','cad-demo']:
        return dispatch(args)
    if args.command=='assembly':
        from .assembly import run
        return 0 if run(args.output,args.viewer,not args.no_render) else 1
    if args.command=='replay':
        from .render import replay
        try: replay(args.run_directory,args.candidate,args.split,args.seed,args.headless)
        except Exception as exc:
            print(f'Replay unavailable ({type(exc).__name__}). Try --headless; on macOS use mjpython -m chassis_lab replay. Physics results are unchanged.',file=sys.stderr)
            return 1
        return 0
    if args.command=='experiment':
        load_dotenv('.env.local',override=False);load_dotenv('.env',override=False)
        if not os.environ.get('OPENAI_API_KEY'):
            p.error('OPENAI_API_KEY is missing. Use demo --offline or configure authorized API credentials.')
    from .experiment import run_experiment
    _,ok=run_experiment(ai=args.command=='experiment',model=args.model,output=args.output,render=not args.no_render)
    return 0 if ok else 1


if __name__=='__main__':
    sys.exit(main())
