from preprocess import Pipeline

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--source_data_dir')
    parser.add_argument('--target_data_dir')
    parser.add_argument('--dataset_name')
    args = parser.parse_args()

    pipeline = Pipeline(args.source_data_dir, args.target_data_dir, args.dataset_name)
    pipeline.run()
