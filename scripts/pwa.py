"""PWA support for Streamlit app."""

import streamlit as st

# Base64 encoded apple-touch-icon (180x180 RAF roundel)
APPLE_TOUCH_ICON = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALQAAAC0CAIAAACyr5FlAAAD3klEQVR42u3dy5FVMQwFQEdAwYosCIbUJimyMgmwAMa2fq06CVjq0tvcZ68v376L/DFLCwQOgUPgEDgEDoFD4BA4BI4+R/35cSpw0DDdygIClM44wkF0hbKYoKQVjkIsShNZWCBSG0cDExWVLCwQKYmjvYzkPhYWiJTBMZBFWiILC0Sy4wAioY9FBh9JcRCQmcgig490OIw8v4+FBSJZcBhwIR+LDD7icRhqOR+LDD4icRhkUR+LDD5icBheaR+LDD5e40jbx/25GuVjTZCx71R7H51x7FcFRxkZO66a+VidZOwc1cbH6iFj56sGPsrj2LkLjhgZu04V9bHI4OMiDizSEpmFY9evQTjIaOyjBo7dsZrjIKO3j+w4dvfqiYOM9j4WGXwUw7HnVRMcZAzxAQccdXDs2VUbBxlzfCTCgcUDHxdxWBujlseyNiyPAzjImOYDDjgS44hq/a+vP/4+zXwcxtFmbfyTiXAlsctjzVkbn2QRRQSOMizeExmKo7SM6j6O4ai7Nq7KeOYjanmsrmvjAYuXROCoKuOBDzgKy7jtIykOa2Ps8ljWhuUxAke4jKs+RuDoLaOWDzjguIOjym9KKhn3fDz+ZYEDjjQ4hsgo5AMOOOCAA44SMi75gAMOOOCAAw444IADDjjggAMOOOCAAw444IADDjjggAMOOOCAAw7fc/ieAw444IDD1+dkwAHHfRz+8eYfb3DA4V/2ZLifw/0ccKTzcfVcGXFYHu4EszysDfeQuoc0Pw43GMfKOIbD3efuPvdqglcTvLfivZUSOLzUFCLjMA5vvIWU1yG9DplxbcABhxepySj9XDkfb2RcxGF5zFkbuXDwkWpt/A8OPobIyIhjso/bjYUDjlAcfEyQkRfHNB8P+vkOBx/tZXwKBx+9ZRTA0dvHmwbG4OCjsYwyOJoRedm0SBx8dJVRD0d1H497FY/jvY+KRN636MBYj+Dgo5+MkzhCfOQnEtKTYwOtjiMtkcBuZMQR6yMPkdgmnJzmWRzhPmKJhJ/98CiP48jg47GSJOc9P8fGOG4ryXbMGjgS+jgFJe25rgzxEo7MPvrl1gTv4eCjtIzrOPioK+MFDj6KyniEg4+KMt7h4KOcjKc4+Kgl4zUOPgrJCMDBRxUZMTgQyc8iGAcfyWUE4+Ajs4x4HIjkZJEIBx8JZSTCwUc2GblwIJKHRVIcY4lknEJOHKOI5O1/ZhwTfKRufnIcjYkUaHsJHJ2UVOp2LRylidTrc0Uc5YhU7XBdHPmVlG9sAxypoLRqZjMcIVDaNrAxjntWpnRsDg6BQ+AQOAQOgUPgEDgEDpma32Dh7kf/MvciAAAAAElFTkSuQmCC"


def inject_pwa_tags():
    """Inject PWA meta tags and manifest link into the page.

    Call this once at the top of each page, after st.set_page_config().
    Uses JavaScript to inject into <head> for iOS compatibility.
    """
    st.markdown(f"""
        <script>
        (function() {{
            if (window._pwaTagsInjected) return;
            window._pwaTagsInjected = true;

            const head = document.head;

            // Apple touch icon with embedded data URL
            const appleIcon = document.createElement('link');
            appleIcon.rel = 'apple-touch-icon';
            appleIcon.href = '{APPLE_TOUCH_ICON}';
            head.appendChild(appleIcon);

            const appleIcon180 = document.createElement('link');
            appleIcon180.rel = 'apple-touch-icon';
            appleIcon180.sizes = '180x180';
            appleIcon180.href = '{APPLE_TOUCH_ICON}';
            head.appendChild(appleIcon180);

            // iOS meta tags
            const metas = [
                ['apple-mobile-web-app-capable', 'yes'],
                ['apple-mobile-web-app-status-bar-style', 'black-translucent'],
                ['apple-mobile-web-app-title', 'Football'],
                ['mobile-web-app-capable', 'yes'],
                ['theme-color', '#0e1117']
            ];
            metas.forEach(([name, content]) => {{
                const meta = document.createElement('meta');
                meta.name = name;
                meta.content = content;
                head.appendChild(meta);
            }});

            // Manifest
            const manifest = document.createElement('link');
            manifest.rel = 'manifest';
            manifest.href = '/app/static/manifest.json';
            head.appendChild(manifest);
        }})();
        </script>
    """, unsafe_allow_html=True)


def show_install_prompt():
    """Show an install prompt for users on mobile browsers."""
    st.markdown("""
        <script>
        let deferredPrompt;

        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;

            const installBanner = document.createElement('div');
            installBanner.id = 'pwa-install-banner';
            installBanner.innerHTML = `
                <div style="
                    position: fixed;
                    bottom: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: linear-gradient(135deg, #1f77b4 0%, #2d8fd8 100%);
                    color: white;
                    padding: 12px 24px;
                    border-radius: 25px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                    z-index: 9999;
                    cursor: pointer;
                    font-family: sans-serif;
                    font-size: 14px;
                    font-weight: 600;
                ">
                    <span style="margin-right: 8px;">📱</span>
                    Install App
                    <span style="margin-left: 12px; opacity: 0.7;" onclick="event.stopPropagation(); this.parentElement.parentElement.remove();">✕</span>
                </div>
            `;

            installBanner.addEventListener('click', async () => {
                if (deferredPrompt) {
                    deferredPrompt.prompt();
                    const { outcome } = await deferredPrompt.userChoice;
                    if (outcome === 'accepted') {
                        installBanner.remove();
                    }
                    deferredPrompt = null;
                }
            });

            document.body.appendChild(installBanner);
        });

        window.addEventListener('appinstalled', () => {
            const banner = document.getElementById('pwa-install-banner');
            if (banner) banner.remove();
        });
        </script>
    """, unsafe_allow_html=True)
