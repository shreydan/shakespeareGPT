from dataset import ShakespeareDataset



if __name__ == '__main__':
    dataset = ShakespeareDataset(block_size=256)
    print(dataset[0])