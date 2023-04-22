ct_images_index = FilesIndex(path='/ct_images_??/*', dirs=True)
ct_images_dataset = Dataset(index = ct_images_index, batch_class=CTImagesBatch)
ct_preprocessed_index = FilesIndex(path='/preprocessed/images/*')
ct_preprocessed_dataset = Dataset(index = ct_preprocessed_index, batch_class=CTImagesBatch)
#        
ct_images_pipeline = ct_preprocessed_dataset.pipeline()
     .load(None, 'blosc')
     .split_to_patches(shape=(64, 64, 64))
#        
ct_masks_ds = Dataset(index = ct_preprocessed_index, batch_class=CTImagesBatch)
ct_masks_pipeline = ct_masks_ds.pipeline().
     .load('/preprocessed/masks', 'blosc')
     .split_to_patches(shape=(64, 64, 64))
#       
full_ds = JointDataset((ct_images_pipeline, ct_masks_pipeline))
full_ds.cv_split([0.8, 0.2])
for i in range(MAX_ITER):
   images, masks = full_ds.train.next_batch(BATCH_SIZE, shuffle=True)
   # обучаем модель, подавая в нее снимки и маски
   for images, masks in full_ds.test.gen_batch(BATCH_SIZE, shuffle=False, one_pass=True):
   # рассчитываем метрики качества модели
