(() => {
  'use strict';
  const zh = document.documentElement.lang.startsWith('zh');
  const say = (a,b) => zh ? a : b;
  const toast = document.querySelector('.toast');
  let toastTimer;
  const announce = text => { toast.textContent=text; toast.hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>toast.hidden=true,2400); };
  async function copy(text) {
    try { await navigator.clipboard.writeText(text); return true; }
    catch (_) {
      const el=document.createElement('textarea');el.value=text;el.style.cssText='position:fixed;opacity:0;left:-9999px';document.body.append(el);el.select();const ok=document.execCommand('copy');el.remove();return ok;
    }
  }
  document.querySelector('.share-button')?.addEventListener('click',async()=>announce(await copy(location.href)?say('页面链接已复制','Link copied'):say('请复制浏览器地址栏中的链接','Copy the address from your browser')));
  document.querySelectorAll('pre:has(code)').forEach(pre=>{
    const wrap=document.createElement('div');wrap.className='code-block';pre.before(wrap);wrap.append(pre);
    const btn=document.createElement('button');btn.className='copy-code';btn.type='button';btn.textContent=say('复制','Copy');btn.setAttribute('aria-label',say('复制代码或原始回答','Copy code or original response'));
    btn.addEventListener('click',async()=>{if(await copy(pre.querySelector('code').textContent)){btn.textContent=say('已复制','Copied');setTimeout(()=>btn.textContent=say('复制','Copy'),1800);}else announce(say('请选中文字后复制','Select and copy the text'));});wrap.append(btn);
  });
  const toc=document.querySelector('.toc'), toggle=document.querySelector('.toc-toggle');
  function closeToc(){toc?.classList.remove('open');toggle?.setAttribute('aria-expanded','false');}
  toggle?.addEventListener('click',()=>{const open=toc.classList.toggle('open');toggle.setAttribute('aria-expanded',String(open));});
  document.querySelector('.mobile-close')?.addEventListener('click',closeToc);
  document.addEventListener('click',e=>{if(toc?.classList.contains('open')&&!toc.contains(e.target)&&!toggle?.contains(e.target))closeToc();});
  const progress=document.querySelector('.progress');
  const sections=[...document.querySelectorAll('.article-section')];
  function updateReading(){
    const max=document.documentElement.scrollHeight-innerHeight;
    if(progress)progress.style.transform='scaleX('+(max>0?scrollY/max:0)+')';
    let current=sections[0]?.id;for(const s of sections){if(s.getBoundingClientRect().top<165)current=s.id;}
    document.querySelectorAll('.toc a[href^="#"]').forEach(a=>a.classList.toggle('active',a.hash==='#'+current));
  }
  let ticking=false;addEventListener('scroll',()=>{if(!ticking){requestAnimationFrame(()=>{updateReading();ticking=false;});ticking=true;}},{passive:true});updateReading();
  const autoplay=document.querySelector('#autoplay');
  if(autoplay)autoplay.checked=!matchMedia('(prefers-reduced-motion: reduce)').matches;
  const videos=[...document.querySelectorAll('video')];
  function visible(v){return Boolean(v.offsetWidth||v.offsetHeight)&&v.getClientRects().length>0;}
  function stop(v){v.dataset.automaticPause='1';v.pause();setTimeout(()=>delete v.dataset.automaticPause,100);}
  function refreshPlayback(){
    videos.forEach(v=>{
      if(!visible(v)){stop(v);return;}
      const b=v.getBoundingClientRect(), clip=v.closest('.demo-card-body')?.getBoundingClientRect();
      const portion=Math.max(0,Math.min(b.bottom,innerHeight,clip?.bottom??Infinity)-Math.max(b.top,70,clip?.top??0))/b.height;
      if(autoplay?.checked&&portion>.45&&!v.dataset.userPaused)v.play().catch(()=>{});
      else if(portion===0)stop(v);
    });
  }
  videos.forEach(v=>{
    v.defaultPlaybackRate=Number(v.dataset.defaultSpeed||2)/Number(v.dataset.baseSpeed||1);v.playbackRate=v.defaultPlaybackRate;v.muted=true;
    v.addEventListener('pause',()=>{if(!v.dataset.automaticPause&&visible(v))v.dataset.userPaused='1';});
    v.addEventListener('play',()=>{delete v.dataset.userPaused;});
    const select=v.closest('.player').querySelector('select');
    select.addEventListener('change',()=>{v.playbackRate=Number(select.value)/Number(v.dataset.baseSpeed||1);});
    v.addEventListener('ratechange',()=>{const value=String(Math.round(v.playbackRate*Number(v.dataset.baseSpeed||1)*100)/100);if([...select.options].some(o=>o.value===value))select.value=value;});
    v.addEventListener('error',()=>{v.closest('.player').classList.add('player-error');});
  });
  const observer=new IntersectionObserver(refreshPlayback,{threshold:[0,.2,.45,.7,1]});videos.forEach(v=>observer.observe(v));
  autoplay?.addEventListener('change',()=>{if(!autoplay.checked)videos.forEach(stop);else{videos.forEach(v=>delete v.dataset.userPaused);refreshPlayback();}});
  document.addEventListener('visibilitychange',()=>document.hidden?videos.forEach(stop):refreshPlayback());
  const tablists=[...document.querySelectorAll('[role=tablist]')];
  function selectTab(button,focus=false){
    const list=button.closest('[role=tablist]');
    list.querySelectorAll(':scope > [role=tab]').forEach(b=>{
      const selected=b===button; b.setAttribute('aria-selected',String(selected));b.tabIndex=selected?0:-1;
      const panel=document.getElementById(b.getAttribute('aria-controls'));if(panel)panel.hidden=!selected;
    });
    if(focus)button.focus();refreshPlayback();updateReading();
  }
  tablists.forEach(list=>{
    const buttons=[...list.querySelectorAll(':scope > [role=tab]')];
    if(buttons[0])selectTab(buttons[0]);
    list.addEventListener('click',e=>{const button=e.target.closest('[role=tab]');if(button&&button.parentElement===list)selectTab(button);});
    list.addEventListener('keydown',e=>{
      const current=buttons.indexOf(document.activeElement);if(current<0)return;
      let next;if(e.key==='ArrowRight')next=(current+1)%buttons.length;else if(e.key==='ArrowLeft')next=(current+buttons.length-1)%buttons.length;else if(e.key==='Home')next=0;else if(e.key==='End')next=buttons.length-1;else return;
      e.preventDefault();selectTab(buttons[next],true);
    });
  });
  const demoControllers=new Map();
  document.querySelectorAll('.demo-gallery').forEach(gallery=>{
    const cards=[...gallery.querySelectorAll(':scope > .demo-card')];
    const group=gallery.closest('.demo-group'), expand=group.querySelector('.demo-expand');
    let selected=0, expanded=false;
    function show(index, focusDirection){
      selected=Math.max(0,Math.min(index,cards.length-1));
      cards.forEach((card,i)=>card.hidden=!expanded&&i!==selected);
      const current=cards[selected];
      if(focusDirection){
        current.querySelector('.demo-card-body').scrollTop=0;
        const control=current.querySelector('.demo-page-'+focusDirection);
        (control.disabled?current.querySelector('.demo-page-'+(focusDirection==='next'?'prev':'next')):control).focus({preventScroll:true});
      }
      refreshPlayback();updateReading();
    }
    gallery.addEventListener('click',event=>{
      const prev=event.target.closest('.demo-page-prev'), next=event.target.closest('.demo-page-next');
      if(prev)show(selected-1,'prev');
      if(next)show(selected+1,'next');
    });
    gallery.addEventListener('keydown',event=>{
      if(!event.target.closest('.demo-pagination')||expanded)return;
      const index=event.key==='ArrowLeft'?selected-1:event.key==='ArrowRight'?selected+1:event.key==='Home'?0:event.key==='End'?cards.length-1:null;
      if(index===null)return;
      event.preventDefault();show(index,index<selected?'prev':'next');
    });
    expand.addEventListener('click',()=>{
      expanded=!expanded;
      gallery.classList.toggle('is-expanded',expanded);
      expand.setAttribute('aria-expanded',String(expanded));
      expand.querySelector('.demo-expand-label').textContent=expanded?say('收起全部示例','Collapse all demos'):say('展开全部示例','Expand all demos');
      show(selected);
    });
    cards.forEach((card,index)=>demoControllers.set(card,()=>show(index)));
    show(0);
  });
  function revealHash(){
    const id=decodeURIComponent(location.hash.slice(1));if(!id)return;let target=document.getElementById(id);if(!target)return;
    const card=target.closest('.demo-card');if(card)demoControllers.get(card)?.();
    const ancestors=[];for(let el=target;el;el=el.parentElement){if(el.matches('[role=tabpanel]'))ancestors.push(el);if(el.tagName==='DETAILS')el.open=true;}
    ancestors.reverse().forEach(panel=>{const b=document.getElementById(panel.getAttribute('aria-labelledby'));if(b)selectTab(b);});
    closeToc();requestAnimationFrame(()=>target.scrollIntoView({block:'start',behavior:'auto'}));
  }
  addEventListener('hashchange',revealHash);revealHash();
  document.querySelectorAll('.chapter-button').forEach(btn=>btn.addEventListener('click',()=>{
    const v=btn.closest('.demo-card')?.querySelector('video');if(!v)return;
    v.currentTime=Number(btn.dataset.seconds)*2/Number(v.dataset.baseSpeed||1);v.play().catch(()=>{});v.scrollIntoView({block:'center',behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'auto':'smooth'});
  }));
  document.querySelectorAll('.sim-play').forEach(btn=>btn.addEventListener('click',()=>{
    const group=[...btn.closest('.demo-card').querySelectorAll('video')];const pause=group.every(v=>!v.paused);
    if(pause){group.forEach(v=>{v.dataset.userPaused='1';v.pause();});btn.textContent=say('一起播放','Play together');}
    else{group.forEach(v=>{v.currentTime=0;delete v.dataset.userPaused;v.play().catch(()=>{});});btn.textContent=say('全部暂停','Pause all');}
  }));
  const dialog=document.querySelector('.image-dialog');
  document.querySelectorAll('.figure-button').forEach(btn=>btn.addEventListener('click',()=>{
    const img=btn.querySelector('img'),target=dialog.querySelector('.dialog-image');target.src=img.src;target.alt=img.alt;dialog.querySelector('.dialog-download').href=img.src;dialog.showModal();
  }));
  dialog?.querySelector('button')?.addEventListener('click',()=>dialog.close());
  dialog?.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
  document.addEventListener('keydown',e=>{if(e.key==='Escape')closeToc();});
  refreshPlayback();
})();
