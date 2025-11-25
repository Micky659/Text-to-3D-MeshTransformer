from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch
import trimesh
from model_scripts.autoencoder import MeshAutoencoder
from model_scripts.transformer import MeshTransformer
from data_scripts.helper_funcs import derive_face_edges_from_faces
from data_scripts.BPT import BPT_serialize


class MeshDataset(Dataset):

    def __init__(self, data):
        self.data = data
        print(f"[MeshDataset] Created from {len(self.data)} entries")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def save(self, path):
        np.savez_compressed(path, self.data, allow_pickle=True)
        print(f"[MeshDataset] Saved {len(self.data)} entries at {path}")

    @classmethod
    def load(cls, path):
        ld = np.load(path, allow_pickle=True)
        d = [item for item in ld["arr_0"]]
        print(f"[MeshDataset] Loaded {len(d)} entries")
        return cls(d)

    def sort_dataset_keys(self):
        order = ['vertices', 'faces', 'face_edges', 'texts', 'text_embeds', 'codes', 'bpt_codes']
        self.data = [{k: d[k] for k in order if k in d} for d in self.data]

    def generate_face_edges(self, batch_size=5):
        dtp = [item for item in self.data if 'face_edges' not in item]
        tb = (len(dtp) + batch_size - 1) // batch_size
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        for i in tqdm(range(0, len(dtp), batch_size), total=tb):
            bd = dtp[i:i + batch_size]

            if not bd:
                continue

            pbf = pad_sequence([item['faces'] for item in bd], batch_first=True, padding_value=-1).to(dev)
            bfe = derive_face_edges_from_faces(pbf, pad_id=-1)
            m = (bfe != -1).all(dim=-1)

            for idx, (ie, im) in enumerate(zip(bfe, m)):
                iem = ie[im]
                bd[idx]['face_edges'] = iem

        self.sort_dataset_keys()
        print(f"[MeshDataset] Generated face_edges for {len(dtp)} entries")

    def generate_codes(self, autoencoder: MeshAutoencoder, batch_size=25):
        tb = (len(self.data) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(self.data), batch_size), total=tb):
            bd = self.data[i:i + batch_size]

            pbv = pad_sequence([item['vertices'] for item in bd], batch_first=True,
                              padding_value=autoencoder.pad_id).to(autoencoder.device)
            pbf = pad_sequence([item['faces'] for item in bd], batch_first=True,
                              padding_value=autoencoder.pad_id).to(autoencoder.device)
            pbfe = pad_sequence([item['face_edges'] for item in bd], batch_first=True,
                               padding_value=autoencoder.pad_id).to(autoencoder.device)

            bc = autoencoder.tokenize(vertices=pbv, faces=pbf, face_edges=pbfe)
            m = (bc != autoencoder.pad_id).all(dim=-1)

            for idx, (ic, im) in enumerate(zip(bc, m)):
                icm = ic[im]
                bd[idx]['codes'] = icm

        self.sort_dataset_keys()
        print(f"[MeshDataset] Generated codes for {len(self.data)} entries")

    def embed_texts(self, transformer: MeshTransformer, batch_size=50):
        ut = list(set(item['texts'] for item in self.data))
        ted = {}

        for i in tqdm(range(0, len(ut), batch_size)):
            bt = ut[i:i + batch_size]
            te = transformer.embed_texts(bt)
            m = (te != transformer.conditioner.text_embed_pad_value).all(dim=-1)

            for idx, t in enumerate(bt):
                me = te[idx][m[idx]]
                ted[t] = me

        for item in self.data:
            if 'texts' in item:
                item['text_embeds'] = ted.get(item['texts'], None)
                del item['texts']

        self.sort_dataset_keys()
        print(f"[MeshDataset] Generated {len(ted)} text_embeddings")

    def generate_codes_bpt(self, batch_size: int = 25, pad_id: int = -1, block_size: int = 16,
                          offset_size: int = 16, use_special_block: bool = False, store_field: str = "bpt_codes"):
        tb = (len(self.data) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(self.data), batch_size), total=tb):
            bd = self.data[i:i + batch_size]

            for item in bd:
                v = item.get('vertices', None)
                f = item.get('faces', None)

                if v is None or f is None:
                    item[store_field] = torch.tensor([], dtype=torch.long)
                    continue

                vn = v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                fn = f.cpu().numpy() if isinstance(f, torch.Tensor) else np.asarray(f)

                if fn.size == 0:
                    item[store_field] = torch.tensor([], dtype=torch.long)
                    continue

                vm = np.all((fn >= 0) & (fn < len(vn)), axis=1)
                fc = fn[vm]

                if fc.size == 0:
                    item[store_field] = torch.tensor([], dtype=torch.long)
                    continue

                try:
                    mesh = trimesh.Trimesh(vertices=vn, faces=fc, process=False)
                    codes = BPT_serialize(mesh, block_size, offset_size, use_special_block)
                except Exception as e:
                    print(f"[generate_codes_bpt] failed with error: {e}")
                    item[store_field] = torch.tensor([], dtype=torch.long)
                    continue

                if isinstance(codes, np.ndarray):
                    item[store_field] = torch.from_numpy(codes.astype(np.int64)).long()
                else:
                    item[store_field] = torch.tensor(codes, dtype=torch.long)

        self.sort_dataset_keys()
        print(f"[MeshDataset] Generated BPT codes for {len(self.data)} entries (pad_id={pad_id})")