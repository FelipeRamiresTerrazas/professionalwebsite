// Single source of truth for CV file metadata used across the site.
(function () {
    var cvConfig = {
        personName: 'Felipe Ramires Terrazas',
        roleTitle: 'Data Specialist',
        fileName: 'cv_felipe_terrazas_data_specialist.pdf',
        rawBaseUrl: 'https://raw.githubusercontent.com/FelipeRamiresTerrazas/cv/main/output/'
    };

    function buildRawUrl(config) {
        return config.rawBaseUrl + config.fileName;
    }

    function buildViewerUrl(rawUrl) {
        return 'https://docs.google.com/viewer?url=' + rawUrl + '&embedded=true';
    }

    function applyCvConfig() {
        var rawUrl = buildRawUrl(cvConfig);
        var viewerUrl = buildViewerUrl(rawUrl);

        document.querySelectorAll('.resume-dl-btn, .cv-download-btn').forEach(function (link) {
            link.href = rawUrl;
            link.setAttribute('download', cvConfig.fileName);
        });

        document.querySelectorAll('.resume-iframe, #cvFrame').forEach(function (frame) {
            frame.src = viewerUrl;
            if (frame.classList.contains('resume-iframe')) {
                frame.setAttribute('aria-label', cvConfig.personName + ' - CV');
            }
            if (frame.id === 'cvFrame') {
                frame.setAttribute('title', cvConfig.personName + ' - ' + cvConfig.roleTitle + ' CV');
            }
        });

        var resumeSubtitle = document.querySelector('.resume-view-sub');
        if (resumeSubtitle) {
            resumeSubtitle.textContent = cvConfig.personName + ' - ' + cvConfig.roleTitle;
        }

        var cvSubtitle = document.querySelector('.cv-subtitle');
        if (cvSubtitle) {
            cvSubtitle.textContent = cvConfig.roleTitle + ' CV';
        }
    }

    window.CV_CONFIG = cvConfig;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', applyCvConfig);
    } else {
        applyCvConfig();
    }
})();