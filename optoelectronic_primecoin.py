"""Primecoin Oracle Lab — native Python, configurable JSON-RPC pool client.

This is an experimental Primecoin-style chain-share miner, not a drop-in
replacement for a consensus-tested Primecoin node/miner. Pool protocol and work
schema are configurable. The pool must provide an 80-byte header (hex) or a
76-byte header prefix, plus multiplier/nonce ranges. It must accept the configured
submission JSON. Validate against your pool's protocol before using real funds.

Run: python primecoin_oracle_custom.py
Dependencies: Python 3.10+ (Tkinter is included with most Windows Python installs).
"""
import base64, json, math, queue, re, threading, time, urllib.error, urllib.parse, urllib.request
import tkinter as tk
from tkinter import ttk, messagebox

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

# --------------------------- Prime-chain oracle ------------------------------
def sha256(b): return __import__('hashlib').sha256(b).digest()
def dsha256(b): return sha256(sha256(b))
def jacobi(a, n):
    if n <= 0 or n % 2 == 0: raise ValueError('Jacobi denominator must be positive and odd')
    a %= n; result = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5): result = -result
        a, n = n, a
        if a % 4 == n % 4 == 3: result = -result
        a %= n
    return result if n == 1 else 0

def ell_test(n):
    """Euler-Lagrange-Lifchitz probable-prime test to base 2."""
    if n < 3 or n % 2 == 0: return n == 2
    j = jacobi(2, n)
    return j != 0 and pow(2, (n - 1) // 2, n) == j % n

def fermat_test(n):
    return n == 2 or (n > 2 and n % 2 == 1 and pow(2, n - 1, n) == 1)

def chain_length(origin, kind, max_len=64):
    """Count ELL-tested Cunningham members: kind=1 (2n+1), kind=2 (2n-1)."""
    n = origin - 1 if kind == 1 else origin + 1
    count = 0
    while count < max_len and n > 2 and n % 2 and ell_test(n):
        count += 1
        n = 2*n + 1 if kind == 1 else 2*n - 1
    return count

class PrimecoinOracle:
    """The proof oracle owns origin construction, chain tests and share checking."""
    def __init__(self, header, multiplier, target_length=6.0):
        if len(header) != 80: raise ValueError('Primecoin block header must be exactly 80 bytes')
        self.header = header
        self.multiplier = int(multiplier)
        if self.multiplier <= 0: raise ValueError('Multiplier must be positive')
        self.header_hash = int.from_bytes(dsha256(header), 'little')
        self.origin = self.header_hash * self.multiplier
        self.target_length = float(target_length)
        self.c1 = self.c2 = self.bi_twin = 0
        self.score = 0.0
    def verify(self):
        # Primecoin's two Cunningham directions share the same chain origin.
        self.c1 = chain_length(self.origin, 1)
        self.c2 = chain_length(self.origin, 2)
        self.bi_twin = min(self.c1, self.c2) + 1 if self.c1 and self.c2 else 0
        self.score = max(float(self.c1), float(self.c2), float(self.bi_twin))
        return self.score >= self.target_length
    def report(self):
        return {'header_hash': f'{self.header_hash:064x}', 'multiplier': self.multiplier,
                'origin': str(self.origin), 'cunningham_first': self.c1,
                'cunningham_second': self.c2, 'bi_twin': self.bi_twin,
                'chain_score': self.score, 'target_length': self.target_length}

# ----------------------------- Custom pool ----------------------------------
def get_path(obj, path):
    cur = obj
    for part in path.split('.'):
        if not part: continue
        if isinstance(cur, list): cur = cur[int(part)]
        else: cur = cur[part]
    return cur

def set_path(obj, path, value):
    parts = path.split('.'); cur = obj
    for p in parts[:-1]: cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    if isinstance(cur, list): cur[int(parts[-1])] = value
    else: cur[parts[-1]] = value

def rpc_call(url, user, password, method, params, timeout=20):
    if urllib.parse.urlparse(url).scheme not in ('http','https'):
        raise ValueError('Pool URL must start with http:// or https://')
    payload = json.dumps({'jsonrpc':'1.0','id':'primecoin-oracle','method':method,'params':params}).encode()
    req = urllib.request.Request(url, payload, {'Content-Type':'application/json'})
    if user or password:
        token = base64.b64encode(f'{user}:{password}'.encode()).decode()
        req.add_header('Authorization', 'Basic '+token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response: body=response.read()
    except urllib.error.HTTPError as e:
        body=e.read()
        if e.code == 401: raise RuntimeError('Pool authentication failed (HTTP 401)')
    except urllib.error.URLError as e: raise RuntimeError(f'Pool connection failed: {e.reason}')
    try: data=json.loads(body)
    except Exception: raise RuntimeError('Pool returned non-JSON data: '+body[:200].decode(errors='replace'))
    if data.get('error'): raise RuntimeError('Pool RPC error: '+json.dumps(data['error']))
    return data.get('result', data)

def render_template(template, values):
    # Replace only named placeholders; preserve the surrounding JSON object braces.
    def sub(match):
        key = match.group(1)
        if key not in values: raise ValueError('Unknown submit-template variable: {' + key + '}')
        return json.dumps(values[key])
    return json.loads(re.sub(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', sub, template))

class PoolConfig:
    def __init__(self, values): self.__dict__.update(values)
    def work_params(self): return json.loads(self.work_params_json)
    def submit_params(self, values): return render_template(self.submit_params_json, values)

class Miner(threading.Thread):
    def __init__(self, cfg, events, stop, camera_oracle=None):
        super().__init__(daemon=True); self.cfg=cfg; self.events=events; self.stop=stop; self.camera_oracle=camera_oracle
        self.hashes=0; self.started=time.time()
    def emit(self, text): self.events.put(('log',text))
    def run(self):
        c=self.cfg
        while not self.stop.is_set():
            try:
                work=rpc_call(c.url,c.user,c.password,c.get_method,c.work_params())
                # Editable response paths let different pool JSON schemas be mapped.
                header_hex=get_path(work,c.header_path)
                header=bytes.fromhex(header_hex.removeprefix('0x'))
                if len(header)==76: header += b'\0'*4
                if len(header)!=80: raise ValueError(f'Configured header field resolved to {len(header)} bytes; expected 76 or 80')
                mstart=int(get_path(work,c.multiplier_start_path)); mend=int(get_path(work,c.multiplier_end_path))
                nstart=int(get_path(work,c.nonce_start_path)); nend=int(get_path(work,c.nonce_end_path))
                target=float(get_path(work,c.target_path)) if c.target_path else float(c.target_length)
                if mend<mstart or nend<nstart: raise ValueError('Pool returned an invalid search range')
                seed=0
                if self.camera_oracle is not None and self.camera_oracle.running:
                    seed,_,_=self.camera_oracle.snapshot()
                self.emit(f'New work: multiplier {mstart}..{mend}, nonce {nstart}..{nend}, chain target {target}; camera seed=0x{seed:016x}')
                found=None
                # Camera-pooled seed selects a complete, non-repeating affine traversal.
                # It changes search order only; it cannot alter proof verification.
                nonce_count=nend-nstart+1
                nonce_offset=seed % nonce_count
                nonce_step=((seed >> 17) | 1) % nonce_count if nonce_count>1 else 1
                if nonce_count>1:
                    while math.gcd(nonce_step,nonce_count)!=1: nonce_step=(nonce_step+2)%nonce_count or 1
                for mult in range(mstart,mend+1):
                    for ni in range(nonce_count):
                        nonce=nstart+((nonce_offset+ni*nonce_step)%nonce_count)
                        if self.stop.is_set(): return
                        hdr=header[:76]+(nonce & 0xffffffff).to_bytes(4,'little')
                        oracle=PrimecoinOracle(hdr,mult,target)
                        self.hashes += 1
                        if oracle.verify():
                            found=(hdr,mult,nonce,oracle.report()); break
                    if found: break
                    if mult % 10 == 0:
                        rate=self.hashes/max(0.001,time.time()-self.started)
                        self.events.put(('stats', (self.hashes,rate,mult)))
                if found:
                    hdr,mult,nonce,report=found
                    vals={'header':hdr.hex(),'header_hex':hdr.hex(),'multiplier':mult,
                          'nonce':nonce,'chain_length':report['chain_score'],
                          'c1':report['cunningham_first'],'c2':report['cunningham_second'],
                          'bi_twin':report['bi_twin'],'proof':report}
                    params=c.submit_params(vals)
                    result=rpc_call(c.url,c.user,c.password,c.submit_method,params)
                    self.events.put(('share',(report,result)))
                else:
                    self.emit('Work range exhausted; requesting fresh work.')
            except Exception as e:
                self.emit('ERROR: '+str(e)); self.stop.wait(max(1.0,float(c.poll_seconds)))
            self.stop.wait(max(0.1,float(c.poll_seconds)))


# ----------------------------- Camera oracle ---------------------------------
class CameraOracle:
    """Turns live camera luminance into a pooled 64-bit search-order seed.

    This does not decide whether a Primecoin share is valid. It only permutes
    the order in which the complete supplied nonce range is searched; every
    candidate is still checked by PrimecoinOracle.
    """
    def __init__(self, index=0, threshold=127, kernel=4, stride=2, operator='POPCOUNT'):
        if cv2 is None or np is None:
            raise RuntimeError('Camera oracle requires OpenCV and NumPy: pip install opencv-python numpy')
        self.index=int(index); self.threshold=int(threshold); self.kernel=int(kernel)
        self.stride=int(stride); self.operator=operator
        self.cap=None; self.seed=0; self.bits=0; self.frame=None
        self.lock=threading.Lock(); self.running=False; self.thread=None
    def start(self):
        if self.running: return
        backend=cv2.CAP_DSHOW if __import__('sys').platform=='win32' else cv2.CAP_ANY
        self.cap=cv2.VideoCapture(self.index,backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        if not self.cap.isOpened():
            self.cap.release(); self.cap=None
            raise RuntimeError(f'Cannot open camera {self.index}')
        self.running=True; self.thread=threading.Thread(target=self._loop,daemon=True); self.thread.start()
    def _pool(self, bits):
        out=[]
        for i in range(0,len(bits),max(1,self.stride)):
            block=bits[i:i+max(1,self.kernel)]
            if not block: continue
            ones=sum(block)
            if self.operator=='AND': b=int(all(block))
            elif self.operator=='OR': b=int(any(block))
            elif self.operator=='XOR': b=ones&1
            elif self.operator=='MAJORITY': b=int(ones>len(block)/2)
            else: b=int(ones/len(block)>=0.5)
            out.append(b)
        value=0
        for i,b in enumerate(out[:64]): value|=(b<<i)
        return value
    def _loop(self):
        try:
            while self.running and self.cap is not None:
                ok,frame=self.cap.read()
                if not ok: time.sleep(.05); continue
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                sample=cv2.resize(gray,(8,8),interpolation=cv2.INTER_AREA)
                bits=[int(int(v)>=self.threshold) for v in sample.flatten()]
                pooled=self._pool(bits)
                preview=cv2.resize(frame,(320,240),interpolation=cv2.INTER_AREA)
                rgb=cv2.cvtColor(preview,cv2.COLOR_BGR2RGB)
                ppm=b'P6 320 240 255\n'+np.ascontiguousarray(rgb,dtype=np.uint8).tobytes()
                with self.lock:
                    self.bits=sum((b<<i) for i,b in enumerate(bits))
                    self.seed=pooled; self.frame=ppm
                time.sleep(.04)
        finally:
            if self.cap is not None: self.cap.release(); self.cap=None
    def snapshot(self):
        with self.lock: return self.seed,self.bits,self.frame
    def stop(self):
        self.running=False
        if self.thread and self.thread.is_alive(): self.thread.join(timeout=1.5)
        if self.cap is not None: self.cap.release(); self.cap=None

# ----------------------------------- GUI -------------------------------------
class App:
    def __init__(self, root):
        self.root=root; root.title('Primecoin Oracle Lab — Custom Pool'); root.geometry('1120x850')
        self.events=queue.Queue(); self.stop=threading.Event(); self.miner=None
        self.vars={}; self.camera=CameraOracle if cv2 is not None else None; self.camera_device=None; self.camera_photo=None
        self._build(); root.after(150,self._poll); root.protocol('WM_DELETE_WINDOW',self.close)
    def field(self,parent,label,key,default,width=35,show=None):
        row=ttk.Frame(parent); row.pack(fill='x',padx=5,pady=2)
        ttk.Label(row,text=label,width=27).pack(side='left')
        v=tk.StringVar(value=str(default)); self.vars[key]=v
        ttk.Entry(row,textvariable=v,width=width,show=show).pack(side='left',fill='x',expand=True)
        return v
    def _build(self):
        outer=ttk.Frame(self.root,padding=8); outer.pack(fill='both',expand=True)
        nb=ttk.Notebook(outer); nb.pack(fill='both',expand=True)
        pool=ttk.Frame(nb,padding=8); mapping=ttk.Frame(nb,padding=8); proof=ttk.Frame(nb,padding=8); camera=ttk.Frame(nb,padding=8)
        nb.add(pool,text='Pool connection'); nb.add(mapping,text='Custom work fields'); nb.add(proof,text='Oracle / submission'); nb.add(camera,text='Camera oracle')
        self.field(pool,'JSON-RPC URL','url','http://127.0.0.1:9912/',60)
        self.field(pool,'Port override (0 = URL port)','port','0')
        self.field(pool,'RPC username','user','')
        self.field(pool,'RPC password','password','',show='*')
        self.field(pool,'Get-work RPC method','get_method','getwork')
        self.field(pool,'Submit-share RPC method','submit_method','submit')
        self.field(pool,'Poll interval seconds','poll_seconds','2')
        self.field(pool,'Request params JSON','work_params_json','[]',60)
        ttk.Label(pool,text='Uses HTTP Basic authentication (RPC username/password). Configure the daemon or pool RPC URL and port here.').pack(anchor='w',padx=6,pady=5)
        self.field(mapping,'Work header path','header_path','header')
        self.field(mapping,'Multiplier start path','multiplier_start_path','multiplier_start')
        self.field(mapping,'Multiplier end path','multiplier_end_path','multiplier_end')
        self.field(mapping,'Nonce start path','nonce_start_path','nonce_start')
        self.field(mapping,'Nonce end path','nonce_end_path','nonce_end')
        self.field(mapping,'Target chain length path (blank = fixed)','target_path','target')
        ttk.Label(mapping,text='Paths are dot-separated JSON paths into the get-work result. Header must be 76 or 80 bytes of hex.').pack(anchor='w',padx=6,pady=8)
        self.field(proof,'Fixed target chain length','target_length','6')
        self.field(proof,'Submit params JSON template','submit_params_json','[{"header":{header},"multiplier":{multiplier},"nonce":{nonce},"chain_length":{chain_length},"proof":{proof}}]',70)
        ttk.Label(proof,text='Template variables: {header}, {header_hex}, {multiplier}, {nonce}, {chain_length}, {c1}, {c2}, {bi_twin}, {proof}. Values are JSON encoded.').pack(anchor='w',padx=6,pady=6)
        self.field(camera,'Camera device index','camera_index','0')
        self.field(camera,'Luminance threshold (0-255)','camera_threshold','127')
        self.field(camera,'Pooling kernel size','camera_kernel','4')
        self.field(camera,'Pooling stride','camera_stride','2')
        ttk.Label(camera,text='Pooling operator:').pack(anchor='w',padx=6,pady=3)
        self.camera_op=tk.StringVar(value='POPCOUNT')
        ttk.Combobox(camera,textvariable=self.camera_op,values=['POPCOUNT','AND','OR','XOR','MAJORITY'],state='readonly').pack(anchor='w',padx=6)
        cam_buttons=ttk.Frame(camera); cam_buttons.pack(anchor='w',padx=6,pady=5)
        ttk.Button(cam_buttons,text='Start camera oracle',command=self.start_camera).pack(side='left',padx=3)
        ttk.Button(cam_buttons,text='Stop camera',command=self.stop_camera).pack(side='left',padx=3)
        self.camera_status=ttk.Label(camera,text='Camera stopped — software-only search order')
        self.camera_status.pack(anchor='w',padx=6,pady=3)
        self.camera_canvas=tk.Canvas(camera,width=320,height=240,bg='black')
        self.camera_canvas.pack(anchor='w',padx=6,pady=5)
        self.camera_image_id=self.camera_canvas.create_image(0,0,anchor='nw')
        self.camera_bits_label=ttk.Label(camera,text='Raw bits: — | Pooled seed: —')
        self.camera_bits_label.pack(anchor='w',padx=6)
        ttk.Label(camera,text='Camera luminance is sampled to 8×8 bits and pooled. The resulting seed permutes nonce traversal; all candidates are still checked by the software Primecoin oracle.').pack(anchor='w',padx=6,pady=6)
        buttons=ttk.Frame(outer); buttons.pack(fill='x',pady=5)
        ttk.Button(buttons,text='Test RPC login',command=self.test_rpc).pack(side='left',padx=4)
        ttk.Button(buttons,text='Start mining',command=self.start).pack(side='left',padx=4)
        ttk.Button(buttons,text='Stop',command=self.halt).pack(side='left',padx=4)
        ttk.Button(buttons,text='Test Oracle',command=self.test).pack(side='left',padx=4)
        self.status=ttk.Label(buttons,text='IDLE'); self.status.pack(side='right',padx=8)
        lf=ttk.LabelFrame(outer,text='Oracle / pool log'); lf.pack(fill='both',expand=True)
        self.log=tk.Text(lf,height=17,font=('Consolas',9)); self.log.pack(fill='both',expand=True,padx=5,pady=5)
        self.log.insert('end','Configure your pool fields and JSON response paths. The pool must provide Primecoin-compatible work.\n')
    def config(self):
        v={k:x.get().strip() for k,x in self.vars.items()}
        if v['port'] and int(v['port'])>0:
            u=urllib.parse.urlparse(v['url']); host=u.hostname
            if not host: raise ValueError('Invalid RPC URL')
            if ':' in host and not host.startswith('['): host='['+host+']'
            scheme=u.scheme or 'http'; path=u.path or '/'
            netloc=f'{host}:{int(v["port"])}'
            v['url']=urllib.parse.urlunparse((scheme,netloc,path,'',u.query,''))
        json.loads(v['work_params_json'])
        # Validate template before networking.
        dummy={'header':'00','header_hex':'00','multiplier':1,'nonce':1,'chain_length':1,
               'c1':1,'c2':1,'bi_twin':1,'proof':{}}
        render_template(v['submit_params_json'],dummy)
        if float(v['poll_seconds'])<0: raise ValueError('Poll interval cannot be negative')
        return PoolConfig(v)
    def test_rpc(self):
        try: cfg=self.config()
        except Exception as e: messagebox.showerror('Configuration error',str(e)); return
        self.status.config(text='TESTING RPC')
        self.log.insert('end',f'Connecting to {cfg.url} using JSON-RPC method {cfg.get_method} ...\n')
        def worker():
            try:
                result=rpc_call(cfg.url,cfg.user,cfg.password,cfg.get_method,cfg.work_params())
                self.events.put(('log','RPC authentication and request succeeded. Response: '+json.dumps(result,indent=2,default=str)))
                self.events.put(('rpc_test','OK'))
            except Exception as e:
                self.events.put(('log','RPC TEST FAILED: '+str(e)))
                self.events.put(('rpc_test','FAILED'))
        threading.Thread(target=worker,daemon=True).start()

    def start_camera(self):
        if self.camera is None:
            messagebox.showerror('Camera unavailable','Install OpenCV and NumPy: pip install opencv-python numpy'); return
        try:
            if self.camera_device is not None and self.camera_device.running: return
            self.camera_device=CameraOracle(int(self.vars['camera_index'].get()),int(self.vars['camera_threshold'].get()),int(self.vars['camera_kernel'].get()),int(self.vars['camera_stride'].get()),self.camera_op.get())
            self.camera_device.start(); self.camera_status.config(text='Camera oracle running')
        except Exception as e: messagebox.showerror('Camera error',str(e))
    def stop_camera(self):
        if self.camera_device is not None: self.camera_device.stop()
        self.camera_status.config(text='Camera stopped — software-only search order')
    def start(self):
        if self.miner and self.miner.is_alive(): return
        try: cfg=self.config()
        except Exception as e: messagebox.showerror('Configuration error',str(e)); return
        self.stop.clear(); self.miner=Miner(cfg,self.events,self.stop,self.camera_device); self.miner.start()
        self.status.config(text='MINING')
    def halt(self):
        self.stop.set(); self.status.config(text='STOPPING')
    def test(self):
        try:
            # Small known demonstration: exercise ELL/Jacobi and Oracle construction.
            assert jacobi(2,7)==1 and jacobi(2,5)==-1
            assert ell_test(7) and ell_test(17) and not ell_test(9)
            hdr=bytes(76)+(0).to_bytes(4,'little')
            o=PrimecoinOracle(hdr,1,6); ok=o.verify()
            self.log.insert('end',f'Oracle self-test passed. Sample chain report: {json.dumps(o.report())}\n')
        except Exception as e: messagebox.showerror('Oracle test failed',str(e))
    def _poll(self):
        try:
            while True:
                kind,data=self.events.get_nowait()
                if kind=='log': self.log.insert('end',data+'\n')
                elif kind=='camera':
                    seed,bits,ppm=data
                    if ppm is not None:
                        self.camera_photo=tk.PhotoImage(data=ppm,format='PPM')
                        self.camera_canvas.itemconfig(self.camera_image_id,image=self.camera_photo)
                    self.camera_bits_label.config(text=f'Raw bits: 0x{bits:016x} | Pooled seed: 0x{seed:016x}')
                elif kind=='stats':
                    h,r,m=data; self.status.config(text=f'{h:,} candidates | {r:,.2f} candidates/s | multiplier {m}')
                elif kind=='rpc_test':
                    self.status.config(text='RPC OK' if data=='OK' else 'RPC FAILED')
                elif kind=='share':
                    report,result=data
                    self.log.insert('end','CHAIN SHARE FOUND\n'+json.dumps(report,indent=2)+'\nPOOL RESPONSE: '+json.dumps(result)+'\n')
                    self.status.config(text='SHARE SUBMITTED — check pool response')
                self.log.see('end')
        except queue.Empty: pass
        if self.camera_device is not None and self.camera_device.running:
            seed,bits,ppm=self.camera_device.snapshot()
            self.events.put(('camera',(seed,bits,ppm)))
        self.root.after(150,self._poll)
    def close(self):
        self.stop.set(); self.stop_camera(); self.root.destroy()

if __name__=='__main__':
    root=tk.Tk(); App(root); root.mainloop()
