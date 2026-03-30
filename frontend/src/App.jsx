import { useState, useCallback, useEffect, useRef } from 'react'
import './App.css'

const API = 'http://localhost:8000'

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [rebuilding, setRebuilding] = useState(false)
  const [error, setError] = useState(null)
  const [statusMessage, setStatusMessage] = useState('')
  const [searched, setSearched] = useState(false)
  const [lightbox, setLightbox] = useState(null)
  const [stats, setStats] = useState(null)
  const [progress, setProgress] = useState(null)
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const pollRef = useRef(null)

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/stats`)
      if (!res.ok) return
      const data = await res.json()
      setStats(data)
    } catch {
      // Ignore stats refresh failures and keep the current UI state.
    }
  }, [])

  useEffect(() => {
    fetchStats()
  }, [fetchStats])

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const startPolling = useCallback(() => {
    stopPolling()
    const tick = async () => {
      try {
        const res = await fetch(`${API}/api/progress`)
        if (!res.ok) return
        const data = await res.json()
        setProgress(data)
        if (!data.is_running) {
          stopPolling()
          setRebuilding(false)
          setProgress(null)
          setStatusMessage('')
          fetchStats()
        }
      } catch { /* ignore */ }
    }
    tick()
    pollRef.current = setInterval(tick, 500)
  }, [stopPolling, fetchStats])

  useEffect(() => () => stopPolling(), [stopPolling])

  const search = useCallback(async () => {
    const trimmed = query.trim()
    if (!trimmed) return

    setLoading(true)
    setError(null)
    setSearched(true)

    try {
      let url = `${API}/api/search?q=${encodeURIComponent(trimmed)}&n=40`
      if (dateFrom) url += `&date_from=${encodeURIComponent(dateFrom)}`
      if (dateTo) url += `&date_to=${encodeURIComponent(dateTo)}`
      const res = await fetch(url)
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data = await res.json()
      setResults(data.results)
    } catch (err) {
      setError(err.message || 'Search failed. Is the backend running?')
      setResults([])
    } finally {
      setLoading(false)
    }
  }, [query, dateFrom, dateTo])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') search()
  }

  const resetDatabase = useCallback(async () => {
    if (!window.confirm('Delete the current vector database and rebuild from the configured photo folder?')) {
      return
    }

    setRebuilding(true)
    setError(null)
    setStatusMessage('Rebuilding vector database\u2026')
    setResults([])
    setSearched(false)
    setLightbox(null)

    try {
      const res = await fetch(`${API}/api/reset`, { method: 'POST' })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      startPolling()
    } catch (err) {
      setError(err.message || 'Reset and rebuild failed.')
      setStatusMessage('')
      setRebuilding(false)
    }
  }, [startPolling])

  const lastRun = stats?.last_index_summary
  const reasonSummary = lastRun?.reason_counts
    ? Object.entries(lastRun.reason_counts)
        .map(([reason, count]) => `${reason}: ${count}`)
        .join(' | ')
    : ''

  return (
    <div className="app">
      <header className="header">
        <h1>Image Search</h1>
        <p>Search your photos using natural language</p>
      </header>

      <div className="search-container">
        <div className="search-box">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe what you're looking for... e.g. dog at the park"
            autoFocus
          />
          <button onClick={search} disabled={loading || !query.trim()}>
            {loading ? 'Searching...' : 'Search'}
          </button>
          <button
            className="secondary-button"
            onClick={resetDatabase}
            disabled={rebuilding}
          >
            {rebuilding ? 'Rebuilding...' : 'Reset DB'}
          </button>
        </div>

        {statusMessage && <div className="status-banner">{statusMessage}</div>}

        {progress && progress.is_running && (
          <div className="progress-container">
            <div className="progress-header">
              <span className="progress-phase">{progress.phase}\u2026</span>
              {progress.total > 0 && (
                <span className="progress-count">{progress.current} / {progress.total}</span>
              )}
            </div>
            <div className="progress-bar-track">
              <div
                className={`progress-bar-fill${progress.total === 0 ? ' indeterminate' : ''}`}
                style={progress.total > 0 ? { width: `${progress.percent}%` } : undefined}
              />
            </div>
            {progress.detail && <div className="progress-detail">{progress.detail}</div>}
          </div>
        )}

        <div className="date-filter-row">
          <label>From <input type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)} /></label>
          <label>To <input type="date" value={dateTo} onChange={e => setDateTo(e.target.value)} /></label>
          {(dateFrom || dateTo) && (
            <button className="clear-dates" onClick={() => { setDateFrom(''); setDateTo('') }}>\u2715 Clear</button>
          )}
        </div>

        {stats && (
          <div className="index-panel">
            <div className="index-panel-row">
              <span>{stats.indexed_images} indexed</span>
              <span>source: {stats.photos_dir}</span>
            </div>
            {lastRun && (
              <div className="index-panel-row muted">
                <span>discovered: {lastRun.total_discovered ?? 0}</span>
                <span>sampled: {lastRun.sampled_images ?? 0}</span>
                <span>kept: {lastRun.kept_images ?? 0}</span>
                <span>skipped: {lastRun.skipped_images ?? 0}</span>
              </div>
            )}
            {reasonSummary && (
              <div className="index-panel-row muted">
                <span>reasons: {reasonSummary}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {searched && !loading && results.length > 0 && (
        <div className="stats-bar">
          <span>{results.length} results for "{query}"</span>
          {stats && <span>{stats.indexed_images} images indexed</span>}
        </div>
      )}

      {loading && (
        <div className="loading">
          <div className="spinner" />
          <p>Searching your photos...</p>
        </div>
      )}

      {!loading && searched && results.length === 0 && !error && (
        <div className="empty-state">
          <div className="icon">🔍</div>
          <p>No results found. Try a different description.</p>
        </div>
      )}

      {!searched && !loading && (
        <div className="empty-state">
          <div className="icon">🖼️</div>
          <p>Type a description to search your photos</p>
          <p style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
            Try: "sunset", "people smiling", "mountains", "food"
          </p>
        </div>
      )}

      {results.length > 0 && (
        <div className="results-grid">
          {results.map((item) => (
            <div
              key={item.id}
              className="image-card"
              onClick={() => setLightbox(item)}
            >
              <img
                src={`${API}${item.url}`}
                alt={item.filename}
                loading="lazy"
              />
              <div className="card-info">
                <div className="filename" title={item.filename}>
                  {item.filename}
                </div>
                <div className="score">
                  {(item.score * 100).toFixed(1)}% match
                </div>
                {(item.folder || item.date_taken || item.date_modified) && (
                  <div className="card-meta">
                    {item.folder && <span title={item.folder}>{item.folder}</span>}
                    {item.date_taken
                      ? <span title="EXIF date taken">{item.date_taken.split(' ')[0]}</span>
                      : item.date_modified && <span title="File modified date">{item.date_modified.split('T')[0]}</span>
                    }
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {lightbox && (
        <div className="lightbox-overlay" onClick={() => setLightbox(null)}>
          <button
            className="lightbox-close"
            onClick={() => setLightbox(null)}
            aria-label="Close"
          >
            ✕
          </button>
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <img
              src={`${API}${lightbox.url}`}
              alt={lightbox.filename}
            />
            <div className="lightbox-meta">
              <h3>{lightbox.filename}</h3>
              {lightbox.relative_path && <p className="meta-path">{lightbox.relative_path}</p>}
              <div className="meta-dates">
                {lightbox.date_taken && (
                  <p><span className="meta-icon">📅</span> <strong>EXIF Date Taken:</strong> {lightbox.date_taken}</p>
                )}
                {lightbox.date_modified && (
                  <p><span className="meta-icon">🗂️</span> <strong>File Modified:</strong> {lightbox.date_modified}</p>
                )}
                {!lightbox.date_taken && !lightbox.date_modified && (
                  <p className="meta-missing">No date information available</p>
                )}
              </div>
              {lightbox.folder && <p><span className="meta-icon">📁</span> {lightbox.folder}</p>}
              {lightbox.tags && <p><span className="meta-icon">🏷️</span> {lightbox.tags}</p>}
              {lightbox.comment && <p><span className="meta-icon">💬</span> {lightbox.comment}</p>}
              {lightbox.camera && <p><span className="meta-icon">📷</span> {lightbox.camera}</p>}
              {lightbox.width > 0 && <p><span className="meta-icon">📐</span> {lightbox.width} × {lightbox.height}</p>}
              {lightbox.gps_lat != null && <p><span className="meta-icon">📍</span> {lightbox.gps_lat}, {lightbox.gps_lon}</p>}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
