if __name__ == '__main__':
    val_data_loader = config.init_obj('val_data_loader', torch.utils.data.dataloader,
                                      dataset=val_dataset,
                                      collate_fn=BatchCollateFn())