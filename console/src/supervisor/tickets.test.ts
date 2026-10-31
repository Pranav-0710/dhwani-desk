import { describe, expect, it, vi } from 'vitest'
import { NO_FILTER, fetchTickets, filterTickets, type StoredTicket } from './tickets'

const ticket = (number: string, department: string, ward: string, urgency: string): StoredTicket => ({
  number,
  department,
  ward,
  urgency,
  location_text: '',
  repeat_complaint: false,
  status: 'open',
  decider_revision: 'keyword-standin-1',
  call_id: `call-${number}`,
  created_at: '2026-10-29T10:00:00+00:00',
})

const tickets = [
  ticket('1001', 'roads', 'w04', 'urgent'),
  ticket('1002', 'water', 'w04', 'soon'),
  ticket('1003', 'roads', 'unknown', 'routine'),
]

describe('filterTickets', () => {
  it('shows everything with no filter', () => {
    expect(filterTickets(tickets, NO_FILTER)).toEqual(tickets)
  })

  it('combines the department, ward and urgency filters', () => {
    expect(filterTickets(tickets, { ...NO_FILTER, department: 'roads' }).map((t) => t.number)).toEqual(['1001', '1003'])
    expect(filterTickets(tickets, { ...NO_FILTER, ward: 'w04', urgency: 'soon' }).map((t) => t.number)).toEqual(['1002'])
    expect(filterTickets(tickets, { department: 'roads', ward: 'unknown', urgency: 'urgent' })).toEqual([])
  })
})

describe('fetchTickets', () => {
  it('sends the console token and returns the tickets', async () => {
    const get = vi.fn(async () => new Response(JSON.stringify(tickets), { status: 200 }))

    await expect(fetchTickets('secret', 50, get)).resolves.toEqual(tickets)
    expect(get).toHaveBeenCalledWith('/api/tickets?limit=50', { headers: { Authorization: 'Bearer secret' } })
  })

  it('reports a failed request', async () => {
    const get = vi.fn(async () => new Response('no', { status: 401 }))

    await expect(fetchTickets('wrong', 50, get)).rejects.toThrow('HTTP 401')
  })
})
