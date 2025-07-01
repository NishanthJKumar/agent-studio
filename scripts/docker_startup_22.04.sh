#!/bin/bash
# Set VNC password if provided
if [ -n "$VNC_PASSWORD" ]; then
    echo -n "$VNC_PASSWORD" > /tmp/.password1
    x11vnc -storepasswd "$(cat /tmp/.password1)" /tmp/.password2
    chmod 644 /tmp/.password*  # Change permissions to be readable
    sed -i 's/^command=x11vnc.*/& -rfbauth \/tmp\/.password2/' /etc/supervisor/conf.d/supervisord.conf
    export VNC_PASSWORD=
fi

# Set additional x11vnc arguments if provided
if [ -n "$X11VNC_ARGS" ]; then
    sed -i "s/^command=x11vnc.*/& ${X11VNC_ARGS}/" /etc/supervisor/conf.d/supervisord.conf
fi

# Set Openbox arguments if provided
if [ -n "$OPENBOX_ARGS" ]; then
    sed -i "s#^command=/usr/bin/openbox\$#& ${OPENBOX_ARGS}#" /etc/supervisor/conf.d/supervisord.conf
fi

USER=ubuntu
HOME=/home/$USER
echo "* Setting up for user: $USER"

# Copy configuration files to the user's home directory
cp -r /root/{.config,.gtkrc-2.0,.asoundrc} "${HOME}" 2>/dev/null || true
chown -R "$USER":"$USER" "${HOME}"
[ -d "/dev/snd" ] && chgrp -R adm /dev/snd

sed -i -e "s|%USER%|$USER|" -e "s|%HOME%|$HOME|" /etc/supervisor/conf.d/supervisord.conf

# Modify VNC server to run as ubuntu user
sed -i 's/^command=x11vnc.*/command=x11vnc -xkb -noshm -display :1 -shared -forever -repeat -capslock/' /etc/supervisor/conf.d/supervisord.conf
sed -i '/^command=x11vnc/a user=ubuntu' /etc/supervisor/conf.d/supervisord.conf

# Set up home folder
if [ ! -x "$HOME/.config/pcmanfm/LXDE/" ]; then
    mkdir -p "$HOME"/.config/pcmanfm/LXDE/
    ln -sf /usr/local/share/doro-lxde-wallpapers/desktop-items-0.conf "$HOME"/.config/pcmanfm/LXDE/
    chown -R "$USER":"$USER" "$HOME"
fi

# Enable SSL if configured
if [ -n "$SSL_PORT" ] && [ -e "/etc/nginx/ssl/nginx.key" ]; then
    echo "* enable SSL"
    sed -i 's|#_SSL_PORT_#\(.*\)443\(.*\)|\1'"$SSL_PORT"'\2|' /etc/nginx/sites-enabled/default
    sed -i 's|#_SSL_PORT_#||' /etc/nginx/sites-enabled/default
fi

# Enable HTTP basic authentication if configured
if [ -n "$HTTP_PASSWORD" ]; then
    echo "* enable HTTP base authentication"
    htpasswd -bc /etc/nginx/.htpasswd "$USER" "$HTTP_PASSWORD"
    sed -i 's|#_HTTP_PASSWORD_#||' /etc/nginx/sites-enabled/default
fi

# Enable relative URL root if configured
if [ -n "$RELATIVE_URL_ROOT" ]; then
    echo "* enable RELATIVE_URL_ROOT: '$RELATIVE_URL_ROOT'"
    sed -i 's|#_RELATIVE_URL_ROOT_||' /etc/nginx/sites-enabled/default
    sed -i 's|_RELATIVE_URL_ROOT_|'"$RELATIVE_URL_ROOT"'|' /etc/nginx/sites-enabled/default
fi

# Update Nginx configuration with dynamic ports
if [ -n "$API_WEB_SOCKET" ]; then
    sed -i "s|proxy_pass http://127.0.0.1:6081;|proxy_pass http://127.0.0.1:$API_WEB_SOCKET;|" /etc/nginx/sites-enabled/default
    sed -i "s/--listen 6081/--listen $API_WEB_SOCKET/" /etc/supervisor/conf.d/supervisord.conf
fi

if [ -n "$API_SOCKET" ]; then
    sed -i "s|proxy_pass http://127.0.0.1:6079;|proxy_pass http://127.0.0.1:$API_SOCKET;|" /etc/nginx/sites-enabled/default
    sed -i "s/PORT = 6079/PORT = $API_SOCKET/" /usr/local/lib/web/backend/run.py
fi

if [ -n "$SERVER_SOCKET" ]; then
    sed -i "s|listen\s\+80\s\+default_server;|listen $SERVER_SOCKET;|" /etc/nginx/sites-enabled/default
else
    sed -i "s|listen\s\+80\s\+default_server;|listen 8080;|" /etc/nginx/sites-enabled/default
fi

if [ -n "$VNC_PORT" ]; then
    # Update supervisord configuration for VNC port
    sed -i "s/-display :1/-display :1 -rfbport $VNC_PORT/" /etc/supervisor/conf.d/supervisord.conf
fi

# Clear sensitive environment variables
PASSWORD=
HTTP_PASSWORD=

# Add agent_server to supervisord under ubuntu user space
cat <<EOF > /etc/supervisor/conf.d/agent_server.conf
[program:agent_server]
command=python3 /home/ubuntu/agent_studio/scripts/agent_server.py
autostart=true
autorestart=true
user=ubuntu
stderr_logfile=/var/log/agent_server.err.log
stdout_logfile=/var/log/agent_server.out.log
environment=HOME="/home/ubuntu",USER="ubuntu",PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin",LOGNAME="ubuntu",DISPLAY=":1.0",DONT_PROMPT_WSL_INSTALL=True
EOF

# Make sure to not bind to a privileged port!
echo NGINX
cat /etc/nginx/sites-enabled/default
echo SUPERVISORD
cat /etc/supervisor/conf.d/supervisord.conf
echo RUN
cat /usr/local/lib/web/backend/run.py

# Start supervisord
exec /bin/tini -- supervisord -n -c /etc/supervisor/supervisord.conf || {
    echo "Supervisord exited with status $?"
    tail -n 50 /var/log/supervisor/supervisord.log
    exit 1
}
